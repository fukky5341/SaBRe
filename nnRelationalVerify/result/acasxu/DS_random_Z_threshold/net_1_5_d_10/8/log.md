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
execution time: IAR + RelationalAnalysis = 0.95 + 1.68 = 2.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5474341
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5433295
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.22 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5474341
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5433295

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5456059
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5473315
time: 0.47 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5195159
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5195159
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5456059
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5473315
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.12
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5195159
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.12
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5195159

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5438801
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5438624
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5408606, upper bound: 27.5458937
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5400004, upper bound: 27.5458931
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5438801
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5438624
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 3, lower bound: -27.5408606, upper bound: 27.5458937
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 3, lower bound: -27.5400004, upper bound: 27.5458931

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5421382
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5415415
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354765, upper bound: 27.5397304
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354765, upper bound: 27.5354765
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360695, upper bound: 27.5419499
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5358926, upper bound: 27.5434920
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5341857, upper bound: 27.5438748
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5381470, upper bound: 27.5382272
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5421382
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5415415
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5354765, upper bound: 27.5397304
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5354765, upper bound: 27.5354765
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5360695, upper bound: 27.5419499
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5358926, upper bound: 27.5434920
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5341857, upper bound: 27.5438748
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -27.5381470, upper bound: 27.5382272

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371100, upper bound: 27.5402238
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333033, upper bound: 27.5394630
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5413973
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5354615
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5382143
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5377064
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5003717, upper bound: 27.5003717
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5003717, upper bound: 27.5003717
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5382008, upper bound: 27.5404624
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378784, upper bound: 27.5400821
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136429, upper bound: 27.5136819
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136429, upper bound: 27.5140841
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339225, upper bound: 27.5402790
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339225, upper bound: 27.5346888
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5085193, upper bound: 27.5089203
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5085193, upper bound: 27.5089203
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5371100, upper bound: 27.5402238
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5333033, upper bound: 27.5394630
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5413973
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5354615
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5382143
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5377064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5003717, upper bound: 27.5003717
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5003717, upper bound: 27.5003717
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5382008, upper bound: 27.5404624
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5378784, upper bound: 27.5400821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5136429, upper bound: 27.5136819
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5136429, upper bound: 27.5140841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5339225, upper bound: 27.5402790
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5339225, upper bound: 27.5346888
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5085193, upper bound: 27.5089203
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 3, lower bound: -27.5085193, upper bound: 27.5089203

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5371883
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5327112
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5366476, upper bound: 27.5391883
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329855, upper bound: 27.5331200
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5329221
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318905, upper bound: 27.5371202
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318905, upper bound: 27.5369110
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5360130
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5358650
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5190883
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5190449
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5184928
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5184928
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.73 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5371883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5327112
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5366476, upper bound: 27.5391883
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5329855, upper bound: 27.5331200
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5329221
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5318905, upper bound: 27.5371202
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5318905, upper bound: 27.5369110
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5360130
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5358650
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5190883
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5190449
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5184928
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.5184928, upper bound: 27.5184928
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.34
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5325141, upper bound: 27.5367136
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5368668
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5297431
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5326382
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5066204, upper bound: 27.5066204
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5066204, upper bound: 27.5066204
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5354879
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5342277
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5354176
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5354795
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5325141, upper bound: 27.5367136
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5368668
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5297431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5326382
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5066204, upper bound: 27.5066204
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5066204, upper bound: 27.5066204
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5354879
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5342277
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5354176
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5354795
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5343520
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5338601
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5295886
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5312213
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5296443
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5341860
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5350037
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5339766
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5321331
time: 0.63 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5343520
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5338601
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5295886
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5312213
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5296443
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5341860
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5350037
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5339766
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.30
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5321331

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.63 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.43
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.63 + 131.22 = 133.85 seconds
