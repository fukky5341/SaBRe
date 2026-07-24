## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 495.41142893616905


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686)
1: (-688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570)
2: (-374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580)
3: (-477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453)
4: (-267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.11 + 2.37 = 3.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -495.4163831, upper bound: 495.4163831

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157273, upper bound: 495.4157273
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157273, upper bound: 495.4157273
time: 1.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 3, lower bound: -495.4157273, upper bound: 495.4157273
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 3, lower bound: -495.4157273, upper bound: 495.4157273

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
time: 0.88 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
time: 1.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.45 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154371

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154370
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154370
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154370
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154370
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
time: 1.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154370
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154370
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154370
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154371, upper bound: 495.4154370
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -495.4154370, upper bound: 495.4154371

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153342, upper bound: 495.4153313
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153341
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153290
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153303
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153313
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153341
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153290
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153303
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153303, upper bound: 495.4153314
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153290, upper bound: 495.4153341
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153341, upper bound: 495.4153314
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153342
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153303, upper bound: 495.4153314
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153290, upper bound: 495.4153341
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153314
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153342
time: 1.01 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153342, upper bound: 495.4153313
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153341
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153290
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153303
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153313
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153341
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153290
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153314, upper bound: 495.4153303
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153303, upper bound: 495.4153314
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153290, upper bound: 495.4153341
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153341, upper bound: 495.4153314
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153342
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153303, upper bound: 495.4153314
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153290, upper bound: 495.4153341
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153314
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 3, lower bound: -495.4153313, upper bound: 495.4153342

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4152759
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4153205
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4152694
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153260
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153205
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4153205, upper bound: 495.4152667
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153232
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4152759
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4152694
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153260
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153205
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152667
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153232
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152694
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153209
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152693
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153209
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152694
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153260
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153209
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152693
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153260
time: 1.05 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4152759
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4153205
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4152694
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153260
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4153205, upper bound: 495.4152667
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153232
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152693, upper bound: 495.4152759
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4152694
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152759, upper bound: 495.4153260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153205
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152667
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152694
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153209
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4152694
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153260
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152759
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4153209
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152694, upper bound: 495.4152693
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -495.4152667, upper bound: 495.4153260

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150782
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150862
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150770, upper bound: 495.4151721
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150782
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150715, upper bound: 495.4150713
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151721
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150770
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150852
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150877, upper bound: 495.4150713
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151854
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150715, upper bound: 495.4150713
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150770
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150852
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
time: 1.15 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150782
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150862
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150770, upper bound: 495.4151721
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150782
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151613
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150715, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151721
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150852
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150877, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151854
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150715, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150770
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150852
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150715
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151582
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150799
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150877
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150713
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4150714
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 3, lower bound: -495.4150713, upper bound: 495.4151629

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145603
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145594
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145677
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145669
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146325
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146314
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145506
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145466
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145529
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146314
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145615
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145520
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145466, upper bound: 495.4145688
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145590
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146335
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145677, upper bound: 495.4145447
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146374
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145603
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145594
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145677
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145669
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146325
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146314
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4146315, upper bound: 495.4145447
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145506
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145466
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145529
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146369
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146314
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145615
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146335
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146374
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146390
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145603
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145594
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145677
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145669
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146264
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146312
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145506
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145466
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145529
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146408
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145615
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145593
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145466, upper bound: 495.4145688
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145668
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146335
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145506
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145465
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145528
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146377
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145603
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145594
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145677
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145669
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146308
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146312
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145506
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145688, upper bound: 495.4145466
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145529
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145520, upper bound: 495.4146351
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146312
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145615
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145593
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145447
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145688
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4145668
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146335
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145447, upper bound: 495.4146315
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686
1: -688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570
2: -374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580
3: -477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453
4: -267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481

Time for backsubstitution: 1.33 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.48 + 417.28 = 420.76 seconds
