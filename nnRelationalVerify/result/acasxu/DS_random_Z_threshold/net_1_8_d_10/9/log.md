## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.859030126


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288)
1: (-45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655)
2: (-21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237)
3: (-39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091)
4: (-28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.98 + 1.88 = 2.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -17.9037896, upper bound: 17.9037896

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9033207, upper bound: 17.9033739
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9033207, upper bound: 17.9033207
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -17.9033207, upper bound: 17.9033739
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -17.9033207, upper bound: 17.9033207

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026215, upper bound: 17.9026647
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026096, upper bound: 17.9026257
time: 0.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029194, upper bound: 17.9029472
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029952, upper bound: 17.9029194
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -17.9026215, upper bound: 17.9026647
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -17.9026096, upper bound: 17.9026257
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -17.9029194, upper bound: 17.9029472
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -17.9029952, upper bound: 17.9029194

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8387655, upper bound: 17.8387737
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8387958, upper bound: 17.8387737
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9017576, upper bound: 17.9018025
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9017576, upper bound: 17.9017809
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027872, upper bound: 17.9028132
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027872, upper bound: 17.9027872
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8959585, upper bound: 17.8959493
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8959585, upper bound: 17.8959493
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.8387655, upper bound: 17.8387737
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.8387958, upper bound: 17.8387737
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.9017576, upper bound: 17.9018025
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.9017576, upper bound: 17.9017809
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.9027872, upper bound: 17.9028132
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.9027872, upper bound: 17.9027872
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.8959585, upper bound: 17.8959493
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 0, lower bound: -17.8959585, upper bound: 17.8959493

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8911402, upper bound: 17.8911402
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8911402, upper bound: 17.8911402
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8932175, upper bound: 17.8932198
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8932175, upper bound: 17.8932198
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027867, upper bound: 17.9027867
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027867, upper bound: 17.9027867
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8524954, upper bound: 17.8524954
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8524954, upper bound: 17.8524954
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8948844, upper bound: 17.8948844
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8948920, upper bound: 17.8948844
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.33 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8911402, upper bound: 17.8911402
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8911402, upper bound: 17.8911402
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8932175, upper bound: 17.8932198
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8932175, upper bound: 17.8932198
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.9027867, upper bound: 17.9027867
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.9027867, upper bound: 17.9027867
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8524954, upper bound: 17.8524954
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8524954, upper bound: 17.8524954
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8948844, upper bound: 17.8948844
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -17.8948920, upper bound: 17.8948844

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7969429, upper bound: 17.7969743
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7969429, upper bound: 17.7969743
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825218
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8893694, upper bound: 17.8893694
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8893694, upper bound: 17.8893694
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8278073, upper bound: 17.8278073
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8278073, upper bound: 17.8278073
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931648
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931627
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931648
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931627
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027761, upper bound: 17.9027761
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9027761, upper bound: 17.9027761
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9012742, upper bound: 17.9012742
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9012742, upper bound: 17.9012742
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8847832, upper bound: 17.8847774
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8847774, upper bound: 17.8847774
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945499, upper bound: 17.8945472
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945472, upper bound: 17.8945472
time: 0.63 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.17 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.7969429, upper bound: 17.7969743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.7969429, upper bound: 17.7969743
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825629
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8825218, upper bound: 17.8825218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8893694, upper bound: 17.8893694
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8893694, upper bound: 17.8893694
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8278073, upper bound: 17.8278073
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8278073, upper bound: 17.8278073
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931648
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931627
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931648
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8931627, upper bound: 17.8931627
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.9027761, upper bound: 17.9027761
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.9027761, upper bound: 17.9027761
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.9012742, upper bound: 17.9012742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.9012742, upper bound: 17.9012742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8847832, upper bound: 17.8847774
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8847774, upper bound: 17.8847774
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8945499, upper bound: 17.8945472
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.17
Output dim: 0, lower bound: -17.8945472, upper bound: 17.8945472

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8823324, upper bound: 17.8823604
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8823324, upper bound: 17.8823324
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8804040, upper bound: 17.8804040
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8804040, upper bound: 17.8804040
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8223242, upper bound: 17.8222913
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8223242, upper bound: 17.8222913
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8887679, upper bound: 17.8887679
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8887679, upper bound: 17.8887679
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8927022
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921828, upper bound: 17.8921828
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921828, upper bound: 17.8921828
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8927022
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8987877, upper bound: 17.8987877
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8987877, upper bound: 17.8987877
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9002941, upper bound: 17.9002941
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9002941, upper bound: 17.9002941
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8718897, upper bound: 17.8718897
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8718897, upper bound: 17.8718897
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8490402, upper bound: 17.8490402
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8490402, upper bound: 17.8490402
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8822086, upper bound: 17.8822086
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8822086, upper bound: 17.8822086
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945163, upper bound: 17.8945163
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8945190, upper bound: 17.8945163
time: 0.54 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.23 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8823324, upper bound: 17.8823604
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8823324, upper bound: 17.8823324
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8804040, upper bound: 17.8804040
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8804040, upper bound: 17.8804040
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8223242, upper bound: 17.8222913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8223242, upper bound: 17.8222913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8887679, upper bound: 17.8887679
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8887679, upper bound: 17.8887679
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8927022
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8921828, upper bound: 17.8921828
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8921828, upper bound: 17.8921828
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8927022
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8926971, upper bound: 17.8926971
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9024984, upper bound: 17.9024984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8987877, upper bound: 17.8987877
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8987877, upper bound: 17.8987877
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9002941, upper bound: 17.9002941
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.9002941, upper bound: 17.9002941
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8718897, upper bound: 17.8718897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8718897, upper bound: 17.8718897
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8490402, upper bound: 17.8490402
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8490402, upper bound: 17.8490402
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8822086, upper bound: 17.8822086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8822086, upper bound: 17.8822086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8945163, upper bound: 17.8945163
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -17.8945190, upper bound: 17.8945163

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819348
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8803618, upper bound: 17.8803618
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8803618, upper bound: 17.8803618
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8765794, upper bound: 17.8765794
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8765794, upper bound: 17.8765794
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878325, upper bound: 17.8878325
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878325, upper bound: 17.8878325
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8477245, upper bound: 17.8477245
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8477245, upper bound: 17.8477245
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918581, upper bound: 17.8918581
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918581, upper bound: 17.8918581
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8907950, upper bound: 17.8907950
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8907950, upper bound: 17.8907950
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9020074, upper bound: 17.9020074
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9020074, upper bound: 17.9020074
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8823118, upper bound: 17.8823118
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8823118, upper bound: 17.8823118
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8799087, upper bound: 17.8799087
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8799087, upper bound: 17.8799087
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914912, upper bound: 17.8914912
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914912, upper bound: 17.8914912
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8978965, upper bound: 17.8978965
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8978965, upper bound: 17.8978965
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8709107, upper bound: 17.8709107
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8709107, upper bound: 17.8709107
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8813285, upper bound: 17.8813285
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8813285, upper bound: 17.8813285
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8766294, upper bound: 17.8766294
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8766294, upper bound: 17.8766294
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8515484, upper bound: 17.8515484
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8515484, upper bound: 17.8515484
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8937459, upper bound: 17.8937459
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8937459, upper bound: 17.8937459
time: 0.71 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.37 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819348
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8819068, upper bound: 17.8819068
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8803618, upper bound: 17.8803618
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8803618, upper bound: 17.8803618
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8765794, upper bound: 17.8765794
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8765794, upper bound: 17.8765794
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8878325, upper bound: 17.8878325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8878325, upper bound: 17.8878325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8477245, upper bound: 17.8477245
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8477245, upper bound: 17.8477245
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918581, upper bound: 17.8918581
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918581, upper bound: 17.8918581
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8907950, upper bound: 17.8907950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8907950, upper bound: 17.8907950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921860, upper bound: 17.8921860
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8918668, upper bound: 17.8918668
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9020074, upper bound: 17.9020074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.9020074, upper bound: 17.9020074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8921061, upper bound: 17.8921061
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8823118, upper bound: 17.8823118
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8823118, upper bound: 17.8823118
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8799087, upper bound: 17.8799087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8799087, upper bound: 17.8799087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8914912, upper bound: 17.8914912
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8914912, upper bound: 17.8914912
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8978965, upper bound: 17.8978965
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8978965, upper bound: 17.8978965
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8709107, upper bound: 17.8709107
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8709107, upper bound: 17.8709107
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8813285, upper bound: 17.8813285
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8813285, upper bound: 17.8813285
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8766294, upper bound: 17.8766294
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8766294, upper bound: 17.8766294
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8515484, upper bound: 17.8515484
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8515484, upper bound: 17.8515484
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8937459, upper bound: 17.8937459
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -17.8937459, upper bound: 17.8937459

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810342
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8713924, upper bound: 17.8713924
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8713924, upper bound: 17.8713924
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7908169, upper bound: 17.7908169
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7908169, upper bound: 17.7908169
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8704663, upper bound: 17.8704663
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8704663, upper bound: 17.8704663
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8760034, upper bound: 17.8760034
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8760034, upper bound: 17.8760034
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8875336, upper bound: 17.8875336
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8875336, upper bound: 17.8875336
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8905984, upper bound: 17.8905984
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8905984, upper bound: 17.8905984
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8239274, upper bound: 17.8239274
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8239274, upper bound: 17.8239274
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914925, upper bound: 17.8914925
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914925, upper bound: 17.8914925
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8757644, upper bound: 17.8757644
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8757644, upper bound: 17.8757644
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8915356, upper bound: 17.8915356
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8915356, upper bound: 17.8915356
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8190258, upper bound: 17.8190258
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8190258, upper bound: 17.8190258
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152425, upper bound: 17.8152425
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152425, upper bound: 17.8152425
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8934030, upper bound: 17.8934030
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8934032, upper bound: 17.8934030
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147
time: 0.70 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.59 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810342
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8713924, upper bound: 17.8713924
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8713924, upper bound: 17.8713924
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8810067, upper bound: 17.8810067
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.7908169, upper bound: 17.7908169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.7908169, upper bound: 17.7908169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695916, upper bound: 17.8695916
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8704663, upper bound: 17.8704663
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8704663, upper bound: 17.8704663
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8760034, upper bound: 17.8760034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8760034, upper bound: 17.8760034
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8875336, upper bound: 17.8875336
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8875336, upper bound: 17.8875336
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8905984, upper bound: 17.8905984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8905984, upper bound: 17.8905984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8239274, upper bound: 17.8239274
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8239274, upper bound: 17.8239274
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8914925, upper bound: 17.8914925
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8914925, upper bound: 17.8914925
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8757644, upper bound: 17.8757644
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8757644, upper bound: 17.8757644
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8915356, upper bound: 17.8915356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8915356, upper bound: 17.8915356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8190258, upper bound: 17.8190258
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8190258, upper bound: 17.8190258
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8152425, upper bound: 17.8152425
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8152425, upper bound: 17.8152425
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8934030, upper bound: 17.8934030
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8934032, upper bound: 17.8934030
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809708
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8788449, upper bound: 17.8788449
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8788449, upper bound: 17.8788449
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8702220, upper bound: 17.8702220
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8702220, upper bound: 17.8702220
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8709385, upper bound: 17.8709385
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8709385, upper bound: 17.8709385
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8698022, upper bound: 17.8698022
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8698022, upper bound: 17.8698022
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8746930, upper bound: 17.8746930
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8746930, upper bound: 17.8746930
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8791439, upper bound: 17.8791439
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8791439, upper bound: 17.8791439
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8875087, upper bound: 17.8875087
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8875087, upper bound: 17.8875087
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8476077, upper bound: 17.8476077
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8476077, upper bound: 17.8476077
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8891503, upper bound: 17.8891503
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8891503, upper bound: 17.8891503
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8898326, upper bound: 17.8898489
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8898326, upper bound: 17.8898326
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914522, upper bound: 17.8914522
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8914522, upper bound: 17.8914522
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8898785, upper bound: 17.8898785
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8898785, upper bound: 17.8898785
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288
1: -45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655
2: -21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237
3: -39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091
4: -28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
time: 0.66 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.52 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809708
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8788449, upper bound: 17.8788449
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8788449, upper bound: 17.8788449
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8702220, upper bound: 17.8702220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8702220, upper bound: 17.8702220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8709385, upper bound: 17.8709385
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8709385, upper bound: 17.8709385
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8809445, upper bound: 17.8809445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8688597, upper bound: 17.8688597
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8691415, upper bound: 17.8691415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8698022, upper bound: 17.8698022
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8698022, upper bound: 17.8698022
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8746930, upper bound: 17.8746930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8746930, upper bound: 17.8746930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8791439, upper bound: 17.8791439
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8791439, upper bound: 17.8791439
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8875087, upper bound: 17.8875087
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8875087, upper bound: 17.8875087
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8476077, upper bound: 17.8476077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8476077, upper bound: 17.8476077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8878223, upper bound: 17.8878223
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8891503, upper bound: 17.8891503
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8891503, upper bound: 17.8891503
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8898326, upper bound: 17.8898489
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8898326, upper bound: 17.8898326
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8888273, upper bound: 17.8888273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8900492, upper bound: 17.8900492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8914522, upper bound: 17.8914522
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8914522, upper bound: 17.8914522
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8752094, upper bound: 17.8752094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8755255, upper bound: 17.8755255
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8898785, upper bound: 17.8898785
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8898785, upper bound: 17.8898785
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8753507, upper bound: 17.8753507
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8903232, upper bound: 17.8903232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8751423, upper bound: 17.8751423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.52
Output dim: 0, lower bound: -17.8749586, upper bound: 17.8749586
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8904688, upper bound: 17.8904688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8753760, upper bound: 17.8753760
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9011869, upper bound: 17.9011869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.9006038, upper bound: 17.9006038
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8917105, upper bound: 17.8917105
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8917664, upper bound: 17.8917664
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8906436, upper bound: 17.8906436
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8820582, upper bound: 17.8820582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8700087, upper bound: 17.8700087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8902379, upper bound: 17.8902379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8897247, upper bound: 17.8897247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8817002, upper bound: 17.8817002
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8695258, upper bound: 17.8695258
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8683615, upper bound: 17.8683615
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8703126, upper bound: 17.8703126
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8803734, upper bound: 17.8803734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8740101, upper bound: 17.8740101
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8750919, upper bound: 17.8750919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8690983, upper bound: 17.8690982
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8934030, upper bound: 17.8934030
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8934032, upper bound: 17.8934030
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -17.8837147, upper bound: 17.8837147

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.86 + 417.64 = 420.50 seconds
