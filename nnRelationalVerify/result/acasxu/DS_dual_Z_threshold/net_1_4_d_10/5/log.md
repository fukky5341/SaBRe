## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 495.22538199974406


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-226.9094849, 344.3358154, -226.9094849, 344.3358154, -571.2452393, 571.2453003)
1: (-253.4875641, 367.0639648, -253.4875641, 367.0639648, -620.5515137, 620.5515137)
2: (-257.4188232, 361.7965088, -257.4188232, 361.7965088, -619.2153320, 619.2153320)
3: (-309.8564758, 425.3124084, -309.8564758, 425.3124084, -735.1688843, 735.1688843)
4: (-281.1933594, 418.6253052, -281.1933594, 418.6253052, -699.8186646, 699.8186646)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 2.53 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -495.2650032, upper bound: 495.2650032

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2304742, upper bound: 495.2304742
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2304742, upper bound: 495.2304742
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -495.2304742, upper bound: 495.2304742
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -495.2304742, upper bound: 495.2304742

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -226.9094849, 344.3358154, -226.9094849, 344.3358154, -571.2452393, 571.2453003
1: -253.4875641, 367.0639648, -253.4875641, 367.0639648, -620.5515137, 620.5515137
2: -257.4188232, 361.7965088, -257.4188232, 361.7965088, -619.2153320, 619.2153320
3: -309.8564758, 425.3124084, -309.8564758, 425.3124084, -735.1688843, 735.1688843
4: -281.1933594, 418.6253052, -281.1933594, 418.6253052, -699.8186646, 699.8186646

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2194531, upper bound: 495.2199216
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2199216, upper bound: 495.2194531
time: 0.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -226.9094849, 344.3358154, -226.9094849, 344.3358154, -571.2452393, 571.2453003
1: -253.4875641, 367.0639648, -253.4875641, 367.0639648, -620.5515137, 620.5515137
2: -257.4188232, 361.7965088, -257.4188232, 361.7965088, -619.2153320, 619.2153320
3: -309.8564758, 425.3124084, -309.8564758, 425.3124084, -735.1688843, 735.1688843
4: -281.1933594, 418.6253052, -281.1933594, 418.6253052, -699.8186646, 699.8186646

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2194531, upper bound: 495.2199216
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2199216, upper bound: 495.2194531
time: 1.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 0, lower bound: -495.2194531, upper bound: 495.2199216
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 0, lower bound: -495.2199216, upper bound: 495.2194531
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 0, lower bound: -495.2194531, upper bound: 495.2199216
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 0, lower bound: -495.2199216, upper bound: 495.2194531

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.16 + 7.67 = 10.83 seconds
