## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.5202488034


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706)
1: (-6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965)
2: (-5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000)
3: (-7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002)
4: (-5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.67 + 1.64 = 2.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.11 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
time: 0.46 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456764, upper bound: 27.5456010
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.69
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.69
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.69
Output dim: 3, lower bound: -27.5456764, upper bound: 27.5456010
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.69
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5441190
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428747
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.86 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5441190
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428747
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.86
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5411491
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5397372
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5344487
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5387729
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5436355
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548
time: 0.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5411491
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5397372
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5344487
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5387729
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5436355
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.71
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.30 + 58.70 = 61.01 seconds
