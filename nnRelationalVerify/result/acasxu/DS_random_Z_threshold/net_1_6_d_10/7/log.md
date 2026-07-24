## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 86.514199010344


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817)
1: (-66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503)
2: (-50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644)
3: (-76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283)
4: (-70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.07 + 2.54 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -86.5211207, upper bound: 86.5211207

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171177, upper bound: 86.5171177
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171177, upper bound: 86.5171177
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -86.5171177, upper bound: 86.5171177
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -86.5171177, upper bound: 86.5171177

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170916, upper bound: 86.5170916
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170916, upper bound: 86.5171177
time: 0.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5120627, upper bound: 86.5120627
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5120627, upper bound: 86.5120627
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -86.5170916, upper bound: 86.5170916
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -86.5170916, upper bound: 86.5171177
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 0, lower bound: -86.5120627, upper bound: 86.5120627
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 0, lower bound: -86.5120627, upper bound: 86.5120627

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154045, upper bound: 86.5154070
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154056, upper bound: 86.5154045
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5168161, upper bound: 86.5169269
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5168161, upper bound: 86.5169292
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.33 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -86.5154045, upper bound: 86.5154070
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -86.5154056, upper bound: 86.5154045
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -86.5168161, upper bound: 86.5169269
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -86.5168161, upper bound: 86.5169292

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5096476, upper bound: 86.5096476
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5096476, upper bound: 86.5096476
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5119788, upper bound: 86.5119788
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5119788, upper bound: 86.5119788
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167977, upper bound: 86.5168640
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167895, upper bound: 86.5169269
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166585, upper bound: 86.5167775
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166254, upper bound: 86.5167592
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5096476, upper bound: 86.5096476
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5096476, upper bound: 86.5096476
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5119788, upper bound: 86.5119788
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5119788, upper bound: 86.5119788
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5167977, upper bound: 86.5168640
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5167895, upper bound: 86.5169269
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5166585, upper bound: 86.5167775
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -86.5166254, upper bound: 86.5167592

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150364, upper bound: 86.5150364
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150364, upper bound: 86.5150364
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166969, upper bound: 86.5168263
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166781, upper bound: 86.5166771
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5060168, upper bound: 86.5060168
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5060168, upper bound: 86.5060167
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167589, upper bound: 86.5166381
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166254, upper bound: 86.5167592
time: 0.70 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.74 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5150364, upper bound: 86.5150364
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5150364, upper bound: 86.5150364
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5166969, upper bound: 86.5168263
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5166781, upper bound: 86.5166771
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5060168, upper bound: 86.5060168
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5060168, upper bound: 86.5060167
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5167589, upper bound: 86.5166381
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 0, lower bound: -86.5166254, upper bound: 86.5167592

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5105460, upper bound: 86.5105460
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5105460, upper bound: 86.5105460
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5117506, upper bound: 86.5118207
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5117506, upper bound: 86.5118207
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143987, upper bound: 86.5144347
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144062, upper bound: 86.5144347
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150735, upper bound: 86.5150735
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150735, upper bound: 86.5150735
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163863, upper bound: 86.5163883
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163863, upper bound: 86.5163863
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165497, upper bound: 86.5166553
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165320, upper bound: 86.5165353
time: 0.94 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.60 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5105460, upper bound: 86.5105460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5105460, upper bound: 86.5105460
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5117506, upper bound: 86.5118207
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5117506, upper bound: 86.5118207
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5143987, upper bound: 86.5144347
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5144062, upper bound: 86.5144347
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5150735, upper bound: 86.5150735
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5150735, upper bound: 86.5150735
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5163863, upper bound: 86.5163883
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5163863, upper bound: 86.5163863
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5165497, upper bound: 86.5166553
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -86.5165320, upper bound: 86.5165353

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5135293, upper bound: 86.5135961
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5135293, upper bound: 86.5135564
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141298, upper bound: 86.5141841
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141517, upper bound: 86.5141608
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5089057, upper bound: 86.5089057
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5089057, upper bound: 86.5089057
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146798, upper bound: 86.5146798
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146798, upper bound: 86.5146798
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163045, upper bound: 86.5163092
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163045, upper bound: 86.5163045
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164060, upper bound: 86.5163752
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163752, upper bound: 86.5163752
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149871, upper bound: 86.5151080
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149870, upper bound: 86.5151080
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
time: 0.81 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.59 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5135293, upper bound: 86.5135961
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5135293, upper bound: 86.5135564
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5141298, upper bound: 86.5141841
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5141517, upper bound: 86.5141608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5089057, upper bound: 86.5089057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5089057, upper bound: 86.5089057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5146798, upper bound: 86.5146798
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5146798, upper bound: 86.5146798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5163045, upper bound: 86.5163092
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5163045, upper bound: 86.5163045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5164060, upper bound: 86.5163752
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5163752, upper bound: 86.5163752
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5149871, upper bound: 86.5151080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5149870, upper bound: 86.5151080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.59
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5108711, upper bound: 86.5108711
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5108711, upper bound: 86.5108711
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145907, upper bound: 86.5145907
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145907, upper bound: 86.5145907
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143412, upper bound: 86.5143412
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143412, upper bound: 86.5143412
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151123, upper bound: 86.5151123
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151123, upper bound: 86.5151123
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5123078, upper bound: 86.5123078
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5123078, upper bound: 86.5123078
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133807, upper bound: 86.5133807
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133807, upper bound: 86.5133807
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149692, upper bound: 86.5150694
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149692, upper bound: 86.5149826
time: 0.80 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5108711, upper bound: 86.5108711
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5108711, upper bound: 86.5108711
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5145907, upper bound: 86.5145907
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5145907, upper bound: 86.5145907
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5143412, upper bound: 86.5143412
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5143412, upper bound: 86.5143412
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5133402, upper bound: 86.5133402
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5151123, upper bound: 86.5151123
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5151123, upper bound: 86.5151123
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5123078, upper bound: 86.5123078
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5123078, upper bound: 86.5123078
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5133807, upper bound: 86.5133807
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5133807, upper bound: 86.5133807
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5149692, upper bound: 86.5150694
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -86.5149692, upper bound: 86.5149826

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137369, upper bound: 86.5137369
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137369, upper bound: 86.5137369
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143385, upper bound: 86.5143385
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143385, upper bound: 86.5143385
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149506, upper bound: 86.5149506
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149506, upper bound: 86.5149506
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146709, upper bound: 86.5146709
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146709, upper bound: 86.5146709
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5076411, upper bound: 86.5076959
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5076411, upper bound: 86.5076959
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133752, upper bound: 86.5133752
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133752, upper bound: 86.5133752
time: 0.98 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.53 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5145808, upper bound: 86.5145808
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5137369, upper bound: 86.5137369
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5137369, upper bound: 86.5137369
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5143385, upper bound: 86.5143385
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5143385, upper bound: 86.5143385
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5149506, upper bound: 86.5149506
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5149506, upper bound: 86.5149506
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5146709, upper bound: 86.5146709
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5146709, upper bound: 86.5146709
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5076411, upper bound: 86.5076959
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5076411, upper bound: 86.5076959
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5133752, upper bound: 86.5133752
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.53
Output dim: 0, lower bound: -86.5133752, upper bound: 86.5133752

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5080304, upper bound: 86.5080304
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5080304, upper bound: 86.5080304
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144113, upper bound: 86.5144113
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144113, upper bound: 86.5144113
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5128055, upper bound: 86.5128055
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5128055, upper bound: 86.5128055
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5139011, upper bound: 86.5139011
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5139011, upper bound: 86.5139011
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137342, upper bound: 86.5137342
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137342, upper bound: 86.5137342
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5136572, upper bound: 86.5136572
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5136572, upper bound: 86.5136572
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133674, upper bound: 86.5133674
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133674, upper bound: 86.5133674
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5111348, upper bound: 86.5111348
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5111348, upper bound: 86.5111348
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5075187, upper bound: 86.5075187
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5075187, upper bound: 86.5075187
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5129910, upper bound: 86.5129910
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5129910, upper bound: 86.5129910
time: 0.67 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.09 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5080304, upper bound: 86.5080304
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5080304, upper bound: 86.5080304
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5144113, upper bound: 86.5144113
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5144113, upper bound: 86.5144113
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5128055, upper bound: 86.5128055
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5128055, upper bound: 86.5128055
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5139011, upper bound: 86.5139011
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5139011, upper bound: 86.5139011
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5137342, upper bound: 86.5137342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5137342, upper bound: 86.5137342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5136572, upper bound: 86.5136572
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5136572, upper bound: 86.5136572
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5133674, upper bound: 86.5133674
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5133674, upper bound: 86.5133674
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5111348, upper bound: 86.5111348
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5111348, upper bound: 86.5111348
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5075187, upper bound: 86.5075187
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5075187, upper bound: 86.5075187
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5129910, upper bound: 86.5129910
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.09
Output dim: 0, lower bound: -86.5129910, upper bound: 86.5129910

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817
1: -66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503
2: -50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644
3: -76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283
4: -70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
time: 0.64 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 2.81 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 2.81
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 2.81
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 2.81
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 2.81
Output dim: 0, lower bound: -86.5101055, upper bound: 86.5101055

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.61 + 152.22 = 155.83 seconds
