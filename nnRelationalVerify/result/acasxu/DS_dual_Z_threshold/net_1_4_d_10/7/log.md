## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 398.85261092052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315)
1: (-197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482)
2: (-197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371)
3: (-234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619)
4: (-201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.59 + 2.58 = 3.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9323974, upper bound: 398.9323973

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8845823
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
time: 1.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8845823
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829353, upper bound: 398.8831322
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832429, upper bound: 398.8828731
time: 1.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828731, upper bound: 398.8832429
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828731, upper bound: 398.8829353
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -398.8829353, upper bound: 398.8831322
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -398.8832429, upper bound: 398.8828731
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -398.8828731, upper bound: 398.8832429
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -398.8828731, upper bound: 398.8829353

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808287, upper bound: 398.8810214
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808859, upper bound: 398.8810401
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810410, upper bound: 398.8807976
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811183, upper bound: 398.8808281
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808281, upper bound: 398.8811183
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8807976, upper bound: 398.8810410
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810401, upper bound: 398.8808859
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810214, upper bound: 398.8808287
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8808287, upper bound: 398.8810214
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8808859, upper bound: 398.8810401
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8810410, upper bound: 398.8807976
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8811183, upper bound: 398.8808281
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8808281, upper bound: 398.8811183
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8807976, upper bound: 398.8810410
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8810401, upper bound: 398.8808859
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -398.8810214, upper bound: 398.8808287

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793983, upper bound: 398.8794467
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8796826
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795786, upper bound: 398.8793678
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793579, upper bound: 398.8797061
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797077, upper bound: 398.8793475
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793678, upper bound: 398.8793547
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798604, upper bound: 398.8793590
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794467, upper bound: 398.8793966
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793966, upper bound: 398.8794467
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8798604
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8793678
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793475, upper bound: 398.8797077
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797061, upper bound: 398.8793579
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8795786
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796826, upper bound: 398.8793590
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794467, upper bound: 398.8793983
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793983, upper bound: 398.8794467
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8796826
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8795786, upper bound: 398.8793678
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793579, upper bound: 398.8797061
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8797077, upper bound: 398.8793475
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793678, upper bound: 398.8793547
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8798604, upper bound: 398.8793590
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8794467, upper bound: 398.8793966
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793966, upper bound: 398.8794467
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8798604
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8793678
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793475, upper bound: 398.8797077
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8797061, upper bound: 398.8793579
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8793590, upper bound: 398.8795786
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8796826, upper bound: 398.8793590
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -398.8794467, upper bound: 398.8793983

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8794409
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8787244
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8793547
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785538, upper bound: 398.8796866
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8795834
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8793328
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781488
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8793436
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8787109
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8792763
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8794409
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8798375
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8795838
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8793547
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793375, upper bound: 398.8781012
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793328, upper bound: 398.8795836
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8793426
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8785538
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8792730
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8793436
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8787109
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8792763
time: 0.87 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8794409
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8787244
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8793547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8785538, upper bound: 398.8796866
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8795834
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8793328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781488
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8793436
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8787109
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8792763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8794409
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8798375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8795838
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8793547
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793375, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793328, upper bound: 398.8795836
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8793426
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8785538
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8792730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8793436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8787109
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8792763

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782376, upper bound: 398.8794409
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8784906
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8785258
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787244
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8785278
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8790527, upper bound: 398.8793547
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8781114
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794350, upper bound: 398.8781012
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796866
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785538, upper bound: 398.8784350
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785432, upper bound: 398.8795834
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8785352
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785674, upper bound: 398.8793328
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8781012
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781117
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781488
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781122
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793692, upper bound: 398.8793436
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8781012
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796622, upper bound: 398.8787109
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8781012
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781042
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788443, upper bound: 398.8792763
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8781629
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781629, upper bound: 398.8794409
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8788443
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781042, upper bound: 398.8781012
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798375
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8796622
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795838
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8793692
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781122, upper bound: 398.8793547
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8781012
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793375, upper bound: 398.8781012
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795836
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793328, upper bound: 398.8785674
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785352, upper bound: 398.8793426
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8785432
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8784350, upper bound: 398.8785538
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8781012
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794350
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781114, upper bound: 398.8792730
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8790527
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785278, upper bound: 398.8793436
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8781012
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785258, upper bound: 398.8787109
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8781012
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8784906, upper bound: 398.8792763
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8782376
time: 0.78 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8782376, upper bound: 398.8794409
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8784906
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8785258
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787244
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8785278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8790527, upper bound: 398.8793547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8781114
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8794350, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796866
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785538, upper bound: 398.8784350
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785432, upper bound: 398.8795834
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8785352
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785674, upper bound: 398.8793328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781117
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781488
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781122
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793692, upper bound: 398.8793436
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8796622, upper bound: 398.8787109
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781042
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8788443, upper bound: 398.8792763
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8781629
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781629, upper bound: 398.8794409
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8788443
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781042, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8796622
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793436, upper bound: 398.8793692
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781122, upper bound: 398.8793547
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793375, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795836
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793328, upper bound: 398.8785674
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785352, upper bound: 398.8793426
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8785432
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8784350, upper bound: 398.8785538
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781114, upper bound: 398.8792730
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8790527
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785278, upper bound: 398.8793436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8785258, upper bound: 398.8787109
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8784906, upper bound: 398.8792763
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -398.8794409, upper bound: 398.8782376

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793823
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794409
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8784906
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796490
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8782499
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785258
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787184
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787244
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8784859
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785278
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793547
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793071
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8781114
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794098, upper bound: 398.8781012
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8781012
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794751, upper bound: 398.8781012
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796866
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796114
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781828
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8784350
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795834
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787883
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785352
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8785287
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793328
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785674, upper bound: 398.8792854
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8781012
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794606, upper bound: 398.8781012
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792854
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781117
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781109
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781488
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781122
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781012
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793436
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8791716
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787852, upper bound: 398.8781012
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8781012
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787109
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798060, upper bound: 398.8781012
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8781012
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8791716
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781042
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792763
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781629
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793823, upper bound: 398.8781012
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793823
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794409
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8788443
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798375
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798060
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796622
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8796523
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795838
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787852
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793692
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8786182
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793547
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8781012
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8781012
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792854, upper bound: 398.8781012
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794606
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795836
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787883
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785674
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785310
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793426
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793059
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787883, upper bound: 398.8781012
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8785432
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785538
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796114, upper bound: 398.8781012
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8781012
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794751
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794350
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794098
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792730
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8790527
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781012
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785278, upper bound: 398.8793436
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8781012
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787184, upper bound: 398.8781012
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8787109
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796490, upper bound: 398.8781012
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8781012
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792763
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8782376
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
time: 0.96 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793823
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794409
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8784906
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793836, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796490
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8782499
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785258
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787184
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787244
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8784859
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793071
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8792730, upper bound: 398.8781114
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8794098, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8794751, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8795329, upper bound: 398.8781012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796866
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796114
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781828
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8784350
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795834
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787883
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785352
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793426, upper bound: 398.8785287
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8785674, upper bound: 398.8792854
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8795836, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8794606, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8796879, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792854
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781117
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781109
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781488
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781122
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793436
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8791716
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8787852, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8795838, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787109
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8798060, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8798375, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793817
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8791716
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781042
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792763
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781629
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793823, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793823
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794409
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8792763, upper bound: 398.8788443
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793817, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8798060
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796622
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8787109, upper bound: 398.8796523
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795838
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787852
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793692
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8786182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793547
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781488, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8792854, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781117, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8796879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795836
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8787883
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785674
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785310
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793426
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793059
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8787883, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8795834, upper bound: 398.8785432
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8785538
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8796114, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8796866, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8795329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794751
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8794098
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792730
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8790527
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8793547, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8785278, upper bound: 398.8793436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8787244, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8787184, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781109, upper bound: 398.8787109
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8796490, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8796667, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8793836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8792763
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8782376
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 0, lower bound: -398.8781012, upper bound: 398.8781012

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8766070
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8766690
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757863
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760431
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757632, upper bound: 398.8757435
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8759960
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8768118
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760957
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8768106
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757439
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8762191, upper bound: 398.8760721
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760772
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8761119
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8762537
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8761256
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8762575
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8759102
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760785
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760785
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8765776
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8765972, upper bound: 398.8757435
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8760818, upper bound: 398.8765767
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8768008, upper bound: 398.8757435
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757522
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8767979, upper bound: 398.8757435
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757437, upper bound: 398.8757435
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8770047, upper bound: 398.8757435
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757719, upper bound: 398.8757435
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8767485, upper bound: 398.8757435
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8769717, upper bound: 398.8757435
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757670, upper bound: 398.8757435
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8759701
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8768839
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760459
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8768089
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757437
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8761103, upper bound: 398.8760149
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760266
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8765667
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8769743
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8761339
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8763567
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8760617
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8757435, upper bound: 398.8757435
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 0.77 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.17 + 416.87 = 420.04 seconds
