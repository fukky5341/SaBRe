## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 9670.419019151372


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438)
1: (-543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422)
2: (-317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418)
3: (-261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383)
4: (-378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.48 + 1.97 = 4.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6124314

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4918847, upper bound: 9670.4918847
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4918847, upper bound: 9670.4918847
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.55 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -9670.4918847, upper bound: 9670.4918847
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -9670.4918847, upper bound: 9670.4918847

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
time: 0.95 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.55 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.55
Output dim: 0, lower bound: -9670.4840255, upper bound: 9670.4840255

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 0, lower bound: -9670.4082909, upper bound: 9670.4082909

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.45 + 25.42 = 29.87 seconds
