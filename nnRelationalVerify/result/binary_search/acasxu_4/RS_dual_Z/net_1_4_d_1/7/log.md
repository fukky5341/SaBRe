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
execution time: IAR + LP analysis = 2.39 + 2.42 = 4.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390144


# Binary Search by BASE starts (time budget: 1195.19 seconds, max iter: 100)

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
Binary search time: 95.10 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1100.09 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8874032
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8874032
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851620, upper bound: 398.8850344
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853707, upper bound: 398.8851501
time: 0.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851501, upper bound: 398.8853707
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850344, upper bound: 398.8851620
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8851620, upper bound: 398.8850344
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8853707, upper bound: 398.8851501
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8851501, upper bound: 398.8853707
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8850344, upper bound: 398.8851620

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849754, upper bound: 398.8850272
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851620, upper bound: 398.8818224
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849515, upper bound: 398.8851495
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853701, upper bound: 398.8849754
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849754, upper bound: 398.8853701
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851495, upper bound: 398.8849515
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818224, upper bound: 398.8851620
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850272, upper bound: 398.8849754
time: 0.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8849754, upper bound: 398.8850272
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8851620, upper bound: 398.8818224
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8849515, upper bound: 398.8851495
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8853701, upper bound: 398.8849754
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8849754, upper bound: 398.8853701
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8851495, upper bound: 398.8849515
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8818224, upper bound: 398.8851620
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 0, lower bound: -398.8850272, upper bound: 398.8849754

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834211, upper bound: 398.8832873
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8833791
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835374, upper bound: 398.8802328
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8805874
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834215, upper bound: 398.8832616
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8835153
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838074, upper bound: 398.8813413
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8834112
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834112, upper bound: 398.8832873
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8838074
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8809374
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8834215
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8832616
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835374
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8813413
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8834211
time: 1.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8834211, upper bound: 398.8832873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8833791
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8835374, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8805874
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8834215, upper bound: 398.8832616
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8835153
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8838074, upper bound: 398.8813413
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8834112
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8834112, upper bound: 398.8832873
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8838074
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8809374
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8834215
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8832616
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835374
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8813413
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8834211

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804148, upper bound: 398.8832873
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834211, upper bound: 398.8803704
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833791
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8804124
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835374, upper bound: 398.8802328
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804127
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813432, upper bound: 398.8832616
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834215, upper bound: 398.8802328
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835153
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8803419
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817284, upper bound: 398.8813413
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838074, upper bound: 398.8802328
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8834112
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804123
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804123, upper bound: 398.8832873
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834112, upper bound: 398.8808799
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8838074
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8817284
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8803419, upper bound: 398.8809374
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8802328
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8834215
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8813432
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804127, upper bound: 398.8832616
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835374
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804124, upper bound: 398.8813413
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8802328
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8803704, upper bound: 398.8834211
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804148
time: 0.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8804148, upper bound: 398.8832873
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8834211, upper bound: 398.8803704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833791
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8804124
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8835374, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804127
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8813432, upper bound: 398.8832616
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8834215, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835153
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8803419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8817284, upper bound: 398.8813413
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8838074, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8834112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8804123, upper bound: 398.8832873
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8834112, upper bound: 398.8808799
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8838074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8817284
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8803419, upper bound: 398.8809374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8834215
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8813432
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8804127, upper bound: 398.8832616
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8804124, upper bound: 398.8813413
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8803704, upper bound: 398.8834211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804148

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804125, upper bound: 398.8832873
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804148, upper bound: 398.8832731
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8806049, upper bound: 398.8802328
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833194, upper bound: 398.8803704
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833791
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833104
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802801
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8804124
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810199, upper bound: 398.8802328
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835204, upper bound: 398.8802328
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8803846
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804127
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804758, upper bound: 398.8832616
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813367, upper bound: 398.8830034
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8806033, upper bound: 398.8802328
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833034, upper bound: 398.8802328
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835036
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8830621
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802492
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8803419
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817123, upper bound: 398.8813413
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817284, upper bound: 398.8802328
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835930, upper bound: 398.8802328
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8837985, upper bound: 398.8802328
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808799, upper bound: 398.8833194
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832744, upper bound: 398.8804114
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804123
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804123, upper bound: 398.8832873
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804114, upper bound: 398.8832744
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8806049, upper bound: 398.8802328
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833194, upper bound: 398.8808799
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8837985
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835930
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8817284
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8817123
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8803419, upper bound: 398.8809374
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802492, upper bound: 398.8802328
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830621, upper bound: 398.8802328
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835036, upper bound: 398.8802328
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833034
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806033
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830034, upper bound: 398.8813367
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804758
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804127, upper bound: 398.8832616
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8803846, upper bound: 398.8802328
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835204
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8810199
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8804124, upper bound: 398.8813413
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802801, upper bound: 398.8802328
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833104, upper bound: 398.8802328
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8802328
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8803704, upper bound: 398.8833194
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832731, upper bound: 398.8804148
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804125
time: 1.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804125, upper bound: 398.8832873
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804148, upper bound: 398.8832731
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8806049, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8833194, upper bound: 398.8803704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833791
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802801
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8804124
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8810199, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8835204, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8805874
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8803846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804127
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804758, upper bound: 398.8832616
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8813367, upper bound: 398.8830034
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8806033, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8833034, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8830621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802492
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8803419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8817123, upper bound: 398.8813413
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8817284, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8835930, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8837985, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8808799, upper bound: 398.8833194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832744, upper bound: 398.8804114
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804123, upper bound: 398.8832873
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804114, upper bound: 398.8832744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8806049, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8833194, upper bound: 398.8808799
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8837985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835930
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8817284
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8817123
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8803419, upper bound: 398.8809374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802492, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8830621, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8835036, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833034
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806033
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8830034, upper bound: 398.8813367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804758
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804127, upper bound: 398.8832616
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8803846, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835204
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8810199
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8804124, upper bound: 398.8813413
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802801, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8833104, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8803704, upper bound: 398.8833194
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832731, upper bound: 398.8804148
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.64
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804125

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8816744
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8814365
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785010, upper bound: 398.8817105
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788377, upper bound: 398.8813555
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8790443, upper bound: 398.8783153
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812169, upper bound: 398.8786760
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817613, upper bound: 398.8783153
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8818573
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8814335
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8817946
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813277
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8784855
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783352
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8787553
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783335
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795839, upper bound: 398.8783153
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8816405, upper bound: 398.8783153
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821846, upper bound: 398.8783153
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789686
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783895
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789716
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8787216
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783365
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814550, upper bound: 398.8787614
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815268, upper bound: 398.8783190
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783190, upper bound: 398.8815268
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788608, upper bound: 398.8814550
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793623, upper bound: 398.8814160
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796852, upper bound: 398.8810137
time: 0.97 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8816744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8814365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8785010, upper bound: 398.8817105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8788377, upper bound: 398.8813555
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8790443, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8812169, upper bound: 398.8786760
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8817613, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8818573
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8814335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8817946
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813277
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8784855
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783352
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8787553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783335
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8795839, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8816405, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8821846, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789686
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8787216
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783365
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8814550, upper bound: 398.8787614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8815268, upper bound: 398.8783190
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8783190, upper bound: 398.8815268
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8788608, upper bound: 398.8814550
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8793623, upper bound: 398.8814160
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.24
Output dim: 0, lower bound: -398.8796852, upper bound: 398.8810137
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8806033, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8833034, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8830621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802492
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8803419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8817123, upper bound: 398.8813413
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8817284, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8835930, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8837985, upper bound: 398.8802328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8808799, upper bound: 398.8833194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8832744, upper bound: 398.8804114
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8804123, upper bound: 398.8832873
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8804114, upper bound: 398.8832744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8806049, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8833194, upper bound: 398.8808799
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8837985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835930
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8817284
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8813413, upper bound: 398.8817123
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8803419, upper bound: 398.8809374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802492, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8830621, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8835036, upper bound: 398.8802328
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8833034
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806033
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8830034, upper bound: 398.8813367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8832616, upper bound: 398.8804758
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8804127, upper bound: 398.8832616
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8803846, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8835204
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8810199
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8804124, upper bound: 398.8813413
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802801, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8833104, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8833791, upper bound: 398.8802328
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8803704, upper bound: 398.8833194
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8806049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8832731, upper bound: 398.8804148
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.24
Output dim: 0, lower bound: -398.8832873, upper bound: 398.8804125
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8865091
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
time: 0.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8865091
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845723, upper bound: 398.8845090
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847436, upper bound: 398.8845686
time: 0.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845686, upper bound: 398.8847436
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845090, upper bound: 398.8845723
time: 1.17 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8845723, upper bound: 398.8845090
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8847436, upper bound: 398.8845686
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8845686, upper bound: 398.8847436
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 0, lower bound: -398.8845090, upper bound: 398.8845723

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8843866
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844487, upper bound: 398.8818224
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841070, upper bound: 398.8844451
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8846132, upper bound: 398.8841138
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8846132
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8841070
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818224, upper bound: 398.8844487
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843866, upper bound: 398.8841138
time: 1.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8843866
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8844487, upper bound: 398.8818224
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8841070, upper bound: 398.8844451
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8846132, upper bound: 398.8841138
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8846132
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8841138, upper bound: 398.8841070
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8818224, upper bound: 398.8844487
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -398.8843866, upper bound: 398.8841138

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828732, upper bound: 398.8829946
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813027, upper bound: 398.8831012
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831807, upper bound: 398.8802328
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829604, upper bound: 398.8805874
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828730, upper bound: 398.8829604
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8831748
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834307, upper bound: 398.8813027
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829946, upper bound: 398.8828724
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828724, upper bound: 398.8829946
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813027, upper bound: 398.8834307
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831748, upper bound: 398.8809374
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829604, upper bound: 398.8828730
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8829604
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8831807
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831012, upper bound: 398.8813027
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829946, upper bound: 398.8828732
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8828732, upper bound: 398.8829946
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8813027, upper bound: 398.8831012
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8831807, upper bound: 398.8802328
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8829604, upper bound: 398.8805874
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8828730, upper bound: 398.8829604
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8809374, upper bound: 398.8831748
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8834307, upper bound: 398.8813027
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8829946, upper bound: 398.8828724
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8828724, upper bound: 398.8829946
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8813027, upper bound: 398.8834307
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8831748, upper bound: 398.8809374
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8829604, upper bound: 398.8828730
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8805874, upper bound: 398.8829604
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8802328, upper bound: 398.8831807
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8831012, upper bound: 398.8813027
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -398.8829946, upper bound: 398.8828732

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8813503
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813047, upper bound: 398.8811461
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8815118
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812120, upper bound: 398.8783153
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815907, upper bound: 398.8783153
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8789716
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8783895
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808661, upper bound: 398.8812202
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813104, upper bound: 398.8811630
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789736, upper bound: 398.8815849
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812073
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813943, upper bound: 398.8783153
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818127, upper bound: 398.8793670
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811622, upper bound: 398.8813020
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8808685
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8813503
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813020, upper bound: 398.8811622
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8818127
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813943
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812073, upper bound: 398.8783153
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815849, upper bound: 398.8789736
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8813104
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8808661
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783895, upper bound: 398.8812202
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789716, upper bound: 398.8811630
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815907
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812120
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8783153
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815118, upper bound: 398.8793670
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8813047
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8808685
time: 1.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8813503
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813047, upper bound: 398.8811461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8815118
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8812120, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8815907, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8789716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8783895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8808661, upper bound: 398.8812202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813104, upper bound: 398.8811630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8789736, upper bound: 398.8815849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812073
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813943, upper bound: 398.8783153
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8818127, upper bound: 398.8793670
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8811622, upper bound: 398.8813020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8808685
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8813503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813020, upper bound: 398.8811622
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8818127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8812073, upper bound: 398.8783153
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8815849, upper bound: 398.8789736
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8813104
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8808661
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783895, upper bound: 398.8812202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8789716, upper bound: 398.8811630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812120
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8815118, upper bound: 398.8793670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8813047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8808685

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785010, upper bound: 398.8813503
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8786760
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788377, upper bound: 398.8811461
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813047, upper bound: 398.8783153
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815118
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8787553
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783352
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812120, upper bound: 398.8783153
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815907, upper bound: 398.8783153
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789716
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8787614
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783895
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8783365
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793824, upper bound: 398.8812202
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808661, upper bound: 398.8783184
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797040, upper bound: 398.8811630
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813104, upper bound: 398.8783153
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815849
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8786613
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812073
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783269
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797782, upper bound: 398.8783153
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813943, upper bound: 398.8783153
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800423, upper bound: 398.8793670
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818127, upper bound: 398.8783153
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787205, upper bound: 398.8813020
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811622, upper bound: 398.8788020
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789971, upper bound: 398.8808685
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8784092
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8784092, upper bound: 398.8813503
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8789971
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786613, upper bound: 398.8811622
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813020, upper bound: 398.8787205
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8818127
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8800423
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813942
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8797782
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783269, upper bound: 398.8783153
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812073, upper bound: 398.8783153
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786613, upper bound: 398.8789736
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783269, upper bound: 398.8783153
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813104
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8797040
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8808661
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8793824
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783365, upper bound: 398.8812202
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783895, upper bound: 398.8783153
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787614, upper bound: 398.8811630
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789716, upper bound: 398.8783153
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815907
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812120
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783352, upper bound: 398.8783153
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8783153
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787553, upper bound: 398.8793670
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815118, upper bound: 398.8783153
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813047
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8788377
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786760, upper bound: 398.8808685
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8785010
time: 1.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8785010, upper bound: 398.8813503
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8786760
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8788377, upper bound: 398.8811461
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813047, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815118
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8793670, upper bound: 398.8787553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783352
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8812120, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8815907, upper bound: 398.8783153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8787614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8783365
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8793824, upper bound: 398.8812202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8808661, upper bound: 398.8783184
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8797040, upper bound: 398.8811630
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813104, upper bound: 398.8783153
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8786613
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812073
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783269
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8797782, upper bound: 398.8783153
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813943, upper bound: 398.8783153
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8800423, upper bound: 398.8793670
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8818127, upper bound: 398.8783153
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8787205, upper bound: 398.8813020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8811622, upper bound: 398.8788020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8789971, upper bound: 398.8808685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8784092
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8784092, upper bound: 398.8813503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8789971
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8786613, upper bound: 398.8811622
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813020, upper bound: 398.8787205
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8818127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8800423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813942
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8797782
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783269, upper bound: 398.8783153
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8812073, upper bound: 398.8783153
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8786613, upper bound: 398.8789736
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783269, upper bound: 398.8783153
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813104
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8797040
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8808661
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8793824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783365, upper bound: 398.8812202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783895, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8787614, upper bound: 398.8811630
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8789716, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8812120
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783352, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8787553, upper bound: 398.8793670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8815118, upper bound: 398.8783153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8811461, upper bound: 398.8788377
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8786760, upper bound: 398.8808685
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -398.8813503, upper bound: 398.8785010

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813305
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8813503
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808685, upper bound: 398.8786760
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788377, upper bound: 398.8810691
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8790443, upper bound: 398.8783153
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813047, upper bound: 398.8783153
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8815118
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8814577
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8784855
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8787553
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8811461
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8810691
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783352
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783335
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812120, upper bound: 398.8783153
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815907, upper bound: 398.8783153
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789686
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8789716
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8787216
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811630, upper bound: 398.8787614
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783895
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783153
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8783153, upper bound: 398.8783365
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812202, upper bound: 398.8783190
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.52 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8849102
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
time: 0.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.23
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8849102
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.23
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829637, upper bound: 398.8829359
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831587, upper bound: 398.8829618
time: 0.91 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829618, upper bound: 398.8831587
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829359, upper bound: 398.8829637
time: 0.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -398.8829637, upper bound: 398.8829359
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -398.8831587, upper bound: 398.8829618
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -398.8829618, upper bound: 398.8831587
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.30
Output dim: 0, lower bound: -398.8829359, upper bound: 398.8829637

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811268, upper bound: 398.8813364
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813681, upper bound: 398.8810907
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811731, upper bound: 398.8813666
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814435, upper bound: 398.8811268
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811268, upper bound: 398.8814435
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813666, upper bound: 398.8811731
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810907, upper bound: 398.8813681
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810907, upper bound: 398.8811276
time: 1.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8811268, upper bound: 398.8813364
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8813681, upper bound: 398.8810907
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8811731, upper bound: 398.8813666
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8814435, upper bound: 398.8811268
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8811268, upper bound: 398.8814435
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8813666, upper bound: 398.8811731
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8810907, upper bound: 398.8813681
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.25
Output dim: 0, lower bound: -398.8810907, upper bound: 398.8811276

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797216, upper bound: 398.8797622
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8799846
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8796639
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8796711
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8796880
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8800208
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8801857, upper bound: 398.8796772
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8797197
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797197, upper bound: 398.8797622
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8801857
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800208, upper bound: 398.8796759
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8798790
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796711, upper bound: 398.8796880
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796639, upper bound: 398.8800231
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796711, upper bound: 398.8796772
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796639, upper bound: 398.8797216
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8797216, upper bound: 398.8797622
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8799846
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8796639
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8796711
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8796880
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8800208
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8801857, upper bound: 398.8796772
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796759, upper bound: 398.8797197
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8797197, upper bound: 398.8797622
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8801857
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8800208, upper bound: 398.8796759
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796880, upper bound: 398.8798790
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796711, upper bound: 398.8796880
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796639, upper bound: 398.8800231
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796711, upper bound: 398.8796772
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 0, lower bound: -398.8796639, upper bound: 398.8797216

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8797559
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797116, upper bound: 398.8781304
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8799712
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8787555
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798751, upper bound: 398.8796529
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800106, upper bound: 398.8781304
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796600
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8781792
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795607, upper bound: 398.8796772
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798317, upper bound: 398.8781304
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786416, upper bound: 398.8800082
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796647, upper bound: 398.8798745
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798778, upper bound: 398.8796660
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8801662, upper bound: 398.8788424
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797097
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8795642
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8797559
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797097, upper bound: 398.8781304
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788424, upper bound: 398.8801662
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8798778
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798745, upper bound: 398.8796647
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800082, upper bound: 398.8786416
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8798317
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8795607
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781792, upper bound: 398.8796772
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796600, upper bound: 398.8781304
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8800106
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796529, upper bound: 398.8798751
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787555, upper bound: 398.8796660
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8799712, upper bound: 398.8788424
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797116
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8795642
time: 0.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8797559
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8797116, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8799712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8787555
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8798751, upper bound: 398.8796529
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8800106, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8781792
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8795607, upper bound: 398.8796772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8798317, upper bound: 398.8781304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8786416, upper bound: 398.8800082
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796647, upper bound: 398.8798745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8798778, upper bound: 398.8796660
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8801662, upper bound: 398.8788424
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8795642
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8797559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8797097, upper bound: 398.8781304
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8788424, upper bound: 398.8801662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8798778
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8798745, upper bound: 398.8796647
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8800082, upper bound: 398.8786416
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8798317
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8795607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781792, upper bound: 398.8796772
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796600, upper bound: 398.8781304
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8800106
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8796529, upper bound: 398.8798751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8787555, upper bound: 398.8796660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8799712, upper bound: 398.8788424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8795642

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797559
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8785215
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781304
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797116, upper bound: 398.8781304
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8799712
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8785564
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8787555
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8785588
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796529
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798751, upper bound: 398.8781304
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781304
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800106, upper bound: 398.8781304
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796600
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781417
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781792
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8781423
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8791718, upper bound: 398.8796772
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795607, upper bound: 398.8781402
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795446, upper bound: 398.8781304
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798317, upper bound: 398.8781304
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8800082
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786416, upper bound: 398.8784665
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786216, upper bound: 398.8798745
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796647, upper bound: 398.8785672
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794863, upper bound: 398.8796660
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798778, upper bound: 398.8781304
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798129, upper bound: 398.8788424
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8801662, upper bound: 398.8781304
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797097
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781339
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789172, upper bound: 398.8795642
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8781974
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781974, upper bound: 398.8797559
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781339, upper bound: 398.8789172
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781339, upper bound: 398.8781304
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797097, upper bound: 398.8781304
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8801662
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788424, upper bound: 398.8798129
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8798778
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8794863
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785672, upper bound: 398.8796647
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798745, upper bound: 398.8786216
time: 1.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797559
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8795642, upper bound: 398.8785215
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8797116, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8799712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8785564
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8787555
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8785588
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796529
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8798751, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8800106, upper bound: 398.8781304
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8796600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781417
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8781423
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8791718, upper bound: 398.8796772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8795607, upper bound: 398.8781402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8795446, upper bound: 398.8781304
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8798317, upper bound: 398.8781304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8800082
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8786416, upper bound: 398.8784665
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8786216, upper bound: 398.8798745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8796647, upper bound: 398.8785672
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8794863, upper bound: 398.8796660
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8798778, upper bound: 398.8781304
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8798129, upper bound: 398.8788424
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8801662, upper bound: 398.8781304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8781339
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8789172, upper bound: 398.8795642
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8781974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781974, upper bound: 398.8797559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781339, upper bound: 398.8789172
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781339, upper bound: 398.8781304
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8797097, upper bound: 398.8781304
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8801662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8788424, upper bound: 398.8798129
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8798778
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8796660, upper bound: 398.8794863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8785672, upper bound: 398.8796647
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.58
Output dim: 0, lower bound: -398.8798745, upper bound: 398.8786216
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8800082, upper bound: 398.8786416
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8798317
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8796772, upper bound: 398.8795607
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8781792, upper bound: 398.8796772
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8796600, upper bound: 398.8781304
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8800106
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8796529, upper bound: 398.8798751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8787555, upper bound: 398.8796660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8799712, upper bound: 398.8788424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8781304, upper bound: 398.8797116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -398.8797559, upper bound: 398.8795642
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1100.37 seconds
