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
execution time: IAR + RelationalAnalysis = 0.88 + 1.82 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6124314

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6124042, upper bound: 9670.6124042
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6124042, upper bound: 9670.6124042
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -9670.6124042, upper bound: 9670.6124042
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -9670.6124042, upper bound: 9670.6124042

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6120004, upper bound: 9670.6120004
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6120004, upper bound: 9670.6120004
time: 0.81 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6122347, upper bound: 9670.6122347
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6122347, upper bound: 9670.6122347
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.98 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -9670.6120004, upper bound: 9670.6120004
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -9670.6120004, upper bound: 9670.6120004
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -9670.6122347, upper bound: 9670.6122347
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -9670.6122347, upper bound: 9670.6122347

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6078490, upper bound: 9670.6078490
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6078490, upper bound: 9670.6078490
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099495, upper bound: 9670.6099495
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099495, upper bound: 9670.6099495
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113583, upper bound: 9670.6113583
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113583, upper bound: 9670.6113583
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6078490, upper bound: 9670.6078490
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6078490, upper bound: 9670.6078490
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6099495, upper bound: 9670.6099495
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6099495, upper bound: 9670.6099495
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6113583, upper bound: 9670.6113583
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6113583, upper bound: 9670.6113583
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6103593, upper bound: 9670.6103593
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6103593, upper bound: 9670.6103593
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4872486, upper bound: 9670.4872486
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4872486, upper bound: 9670.4872486
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6063655, upper bound: 9670.6063655
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6063655, upper bound: 9670.6063655
time: 0.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6103593, upper bound: 9670.6103593
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6103593, upper bound: 9670.6103593
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.4872486, upper bound: 9670.4872486
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.4872486, upper bound: 9670.4872486
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6113061, upper bound: 9670.6113061
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6063655, upper bound: 9670.6063655
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -9670.6063655, upper bound: 9670.6063655

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068853, upper bound: 9670.6068853
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068853, upper bound: 9670.6068853
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064801, upper bound: 9670.6064801
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064801, upper bound: 9670.6064801
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6097221, upper bound: 9670.6097221
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6097221, upper bound: 9670.6097221
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6081389, upper bound: 9670.6081389
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6081389, upper bound: 9670.6081389
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4844326, upper bound: 9670.4844326
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4844326, upper bound: 9670.4844326
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6026713, upper bound: 9670.6026713
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6026713, upper bound: 9670.6026713
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6076651, upper bound: 9670.6076651
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068853, upper bound: 9670.6068853
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068853, upper bound: 9670.6068853
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6068876, upper bound: 9670.6068876
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6090726, upper bound: 9670.6090726
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6064801, upper bound: 9670.6064801
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6064801, upper bound: 9670.6064801
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6097221, upper bound: 9670.6097221
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6097221, upper bound: 9670.6097221
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6081389, upper bound: 9670.6081389
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6081389, upper bound: 9670.6081389
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4844326, upper bound: 9670.4844326
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4844326, upper bound: 9670.4844326
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.4124436, upper bound: 9670.4124436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6026713, upper bound: 9670.6026713
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.6026713, upper bound: 9670.6026713
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068197
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6066250, upper bound: 9670.6066250
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6066250, upper bound: 9670.6066250
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089873, upper bound: 9670.6089873
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089873, upper bound: 9670.6089873
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090410, upper bound: 9670.6090410
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6090410, upper bound: 9670.6090410
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6024369, upper bound: 9670.6024369
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6024369, upper bound: 9670.6024369
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4479173, upper bound: 9670.4479173
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4479173, upper bound: 9670.4479173
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6096602, upper bound: 9670.6096602
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6096602, upper bound: 9670.6096602
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067808, upper bound: 9670.6067808
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067808, upper bound: 9670.6067808
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6071907, upper bound: 9670.6071907
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6071907, upper bound: 9670.6071907
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4479174, upper bound: 9670.4479174
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4479174, upper bound: 9670.4479174
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4316784, upper bound: 9670.4316784
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4316784, upper bound: 9670.4316784
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6015464, upper bound: 9670.6015464
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6015464, upper bound: 9670.6015464
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5878341, upper bound: 9670.5878341
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5878341, upper bound: 9670.5878341
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.56 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6068584, upper bound: 9670.6068584
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068197
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6066250, upper bound: 9670.6066250
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6066250, upper bound: 9670.6066250
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6089873, upper bound: 9670.6089873
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6089873, upper bound: 9670.6089873
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6090410, upper bound: 9670.6090410
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6090410, upper bound: 9670.6090410
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6024369, upper bound: 9670.6024369
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6024369, upper bound: 9670.6024369
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4479173, upper bound: 9670.4479173
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4479173, upper bound: 9670.4479173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6096602, upper bound: 9670.6096602
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6096602, upper bound: 9670.6096602
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6067808, upper bound: 9670.6067808
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6067808, upper bound: 9670.6067808
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6071907, upper bound: 9670.6071907
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6071907, upper bound: 9670.6071907
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4479174, upper bound: 9670.4479174
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4479174, upper bound: 9670.4479174
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4316784, upper bound: 9670.4316784
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4316784, upper bound: 9670.4316784
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6015464, upper bound: 9670.6015464
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.6015464, upper bound: 9670.6015464
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5878341, upper bound: 9670.5878341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5878341, upper bound: 9670.5878341
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5795952, upper bound: 9670.5795952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4006941, upper bound: 9670.4006941
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4006941, upper bound: 9670.4006941
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6059806, upper bound: 9670.6059806
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6059806, upper bound: 9670.6059806
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067979, upper bound: 9670.6067979
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067979, upper bound: 9670.6067979
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064852, upper bound: 9670.6064852
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6064852, upper bound: 9670.6064852
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6021084, upper bound: 9670.6021084
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6021084, upper bound: 9670.6021084
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3996643, upper bound: 9670.3996643
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3996643, upper bound: 9670.3996643
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3679162, upper bound: 9670.3679162
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3679162, upper bound: 9670.3679162
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4100851, upper bound: 9670.4100851
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4100851, upper bound: 9670.4100851
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
time: 0.52 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4006941, upper bound: 9670.4006941
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4006941, upper bound: 9670.4006941
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6064603, upper bound: 9670.6064603
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6059806, upper bound: 9670.6059806
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6059806, upper bound: 9670.6059806
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062170, upper bound: 9670.6062170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6067979, upper bound: 9670.6067979
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6067979, upper bound: 9670.6067979
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6068198, upper bound: 9670.6068198
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6064852, upper bound: 9670.6064852
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6064852, upper bound: 9670.6064852
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6021084, upper bound: 9670.6021084
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6021084, upper bound: 9670.6021084
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.3996643, upper bound: 9670.3996643
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.3996643, upper bound: 9670.3996643
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.3679162, upper bound: 9670.3679162
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.3679162, upper bound: 9670.3679162
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4100851, upper bound: 9670.4100851
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4100851, upper bound: 9670.4100851
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6040295, upper bound: 9670.6040295
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6040295, upper bound: 9670.6040295
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6060636, upper bound: 9670.6060636
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6060636, upper bound: 9670.6060636
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6021925, upper bound: 9670.6021925
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6021925, upper bound: 9670.6021925
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057494, upper bound: 9670.6057494
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057494, upper bound: 9670.6057494
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6044572, upper bound: 9670.6044572
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6044572, upper bound: 9670.6044572
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022194, upper bound: 9670.6022194
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022194, upper bound: 9670.6022194
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6040101, upper bound: 9670.6040101
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6040101, upper bound: 9670.6040101
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5991220, upper bound: 9670.5991220
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5991220, upper bound: 9670.5991220
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6060681, upper bound: 9670.6060681
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6060681, upper bound: 9670.6060681
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6052650, upper bound: 9670.6052650
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6052650, upper bound: 9670.6052650
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062954, upper bound: 9670.6062954
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062954, upper bound: 9670.6062954
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061538, upper bound: 9670.6061538
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061538, upper bound: 9670.6061538
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057671, upper bound: 9670.6057671
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057671, upper bound: 9670.6057671
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6002511, upper bound: 9670.6002511
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6002511, upper bound: 9670.6002511
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6000499, upper bound: 9670.6000499
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6000499, upper bound: 9670.6000499
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067639, upper bound: 9670.6067639
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6067639, upper bound: 9670.6067639
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6065640, upper bound: 9670.6065640
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6065640, upper bound: 9670.6065640
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055427, upper bound: 9670.6055427
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055427, upper bound: 9670.6055427
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061674, upper bound: 9670.6061674
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061674, upper bound: 9670.6061674
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6081399, upper bound: 9670.6081399
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6081399, upper bound: 9670.6081399
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6047771, upper bound: 9670.6047771
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6047771, upper bound: 9670.6047771
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5362375, upper bound: 9670.5362375
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5362375, upper bound: 9670.5362375
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6043941, upper bound: 9670.6043941
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6043941, upper bound: 9670.6043941
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6054967, upper bound: 9670.6054967
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6054967, upper bound: 9670.6054967
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6017455, upper bound: 9670.6017455
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6017455, upper bound: 9670.6017455
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6015672, upper bound: 9670.6015672
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6015672, upper bound: 9670.6015672
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055305, upper bound: 9670.6055305
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6055305, upper bound: 9670.6055305
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3901638, upper bound: 9670.3901638
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3901638, upper bound: 9670.3901638
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3951832, upper bound: 9670.3951832
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3951832, upper bound: 9670.3951832
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4432333, upper bound: 9670.4432333
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4432333, upper bound: 9670.4432333
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4132711, upper bound: 9670.4132711
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.4132711, upper bound: 9670.4132711
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4373456, upper bound: 9670.4373456
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4373456, upper bound: 9670.4373456
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4471346, upper bound: 9670.4471346
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4471346, upper bound: 9670.4471346
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438
1: -543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422
2: -317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418
3: -261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383
4: -378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
time: 0.59 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6040295, upper bound: 9670.6040295
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6040295, upper bound: 9670.6040295
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6046060, upper bound: 9670.6046060
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6060636, upper bound: 9670.6060636
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6060636, upper bound: 9670.6060636
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6021925, upper bound: 9670.6021925
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6021925, upper bound: 9670.6021925
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057494, upper bound: 9670.6057494
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057494, upper bound: 9670.6057494
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6044572, upper bound: 9670.6044572
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6044572, upper bound: 9670.6044572
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022194, upper bound: 9670.6022194
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022194, upper bound: 9670.6022194
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6058065, upper bound: 9670.6058065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055676, upper bound: 9670.6055676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6040101, upper bound: 9670.6040101
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6040101, upper bound: 9670.6040101
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.5991220, upper bound: 9670.5991220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.5991220, upper bound: 9670.5991220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6050422, upper bound: 9670.6050422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6060681, upper bound: 9670.6060681
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6060681, upper bound: 9670.6060681
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6052650, upper bound: 9670.6052650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6052650, upper bound: 9670.6052650
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062954, upper bound: 9670.6062954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062954, upper bound: 9670.6062954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061538, upper bound: 9670.6061538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061538, upper bound: 9670.6061538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057671, upper bound: 9670.6057671
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057671, upper bound: 9670.6057671
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061240, upper bound: 9670.6061240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6002511, upper bound: 9670.6002511
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6002511, upper bound: 9670.6002511
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6000499, upper bound: 9670.6000499
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6000499, upper bound: 9670.6000499
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6009043, upper bound: 9670.6009043
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6067639, upper bound: 9670.6067639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6067639, upper bound: 9670.6067639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6065640, upper bound: 9670.6065640
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6065640, upper bound: 9670.6065640
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055427, upper bound: 9670.6055427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055427, upper bound: 9670.6055427
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061674, upper bound: 9670.6061674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061674, upper bound: 9670.6061674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061062, upper bound: 9670.6061062
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6062401, upper bound: 9670.6062401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6081399, upper bound: 9670.6081399
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6081399, upper bound: 9670.6081399
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6089555, upper bound: 9670.6089555
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6047771, upper bound: 9670.6047771
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6047771, upper bound: 9670.6047771
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6084177, upper bound: 9670.6084177
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.5362375, upper bound: 9670.5362375
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.5362375, upper bound: 9670.5362375
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6043941, upper bound: 9670.6043941
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6043941, upper bound: 9670.6043941
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6054967, upper bound: 9670.6054967
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6054967, upper bound: 9670.6054967
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055212, upper bound: 9670.6055212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6017455, upper bound: 9670.6017455
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6017455, upper bound: 9670.6017455
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6015672, upper bound: 9670.6015672
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6015672, upper bound: 9670.6015672
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6022049, upper bound: 9670.6022049
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6061118, upper bound: 9670.6061118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055305, upper bound: 9670.6055305
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6055305, upper bound: 9670.6055305
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.3901638, upper bound: 9670.3901638
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.3901638, upper bound: 9670.3901638
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.6057788, upper bound: 9670.6057788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.3951832, upper bound: 9670.3951832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.3951832, upper bound: 9670.3951832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4348903, upper bound: 9670.4348903
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4261925, upper bound: 9670.4261925
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4507920, upper bound: 9670.4507920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4380135, upper bound: 9670.4380135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4432333, upper bound: 9670.4432333
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4432333, upper bound: 9670.4432333
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4132711, upper bound: 9670.4132711
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4132711, upper bound: 9670.4132711
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4373456, upper bound: 9670.4373456
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4373456, upper bound: 9670.4373456
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4448345, upper bound: 9670.4448345
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4400753, upper bound: 9670.4400753
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4471346, upper bound: 9670.4471346
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4471346, upper bound: 9670.4471346
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -9670.4559045, upper bound: 9670.4559045
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6093087, upper bound: 9670.6093087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6078562, upper bound: 9670.6078562
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6036968, upper bound: 9670.6036968
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6052490, upper bound: 9670.6052490
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6066884, upper bound: 9670.6066884
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6058540, upper bound: 9670.6058540
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6071264, upper bound: 9670.6071264
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4465333, upper bound: 9670.4465333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4525151, upper bound: 9670.4525151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4415360, upper bound: 9670.4415360
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4464558, upper bound: 9670.4464558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4510556, upper bound: 9670.4510556
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5814504, upper bound: 9670.5814504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.6000218, upper bound: 9670.6000218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5860376, upper bound: 9670.5860376
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5419981, upper bound: 9670.5419981
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.4216916, upper bound: 9670.4216916
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5763136, upper bound: 9670.5763136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 0, lower bound: -9670.5764033, upper bound: 9670.5764033

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.70 + 417.60 = 420.29 seconds
