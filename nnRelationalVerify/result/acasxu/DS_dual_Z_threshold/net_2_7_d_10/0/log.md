## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 19178.25882359392


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469)
1: (-1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926)
2: (-636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629)
3: (-516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145)
4: (-748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.38 + 2.07 = 4.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -19178.4506081, upper bound: 19178.4506081

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
time: 0.69 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
time: 0.66 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.75
Output dim: 0, lower bound: -19178.4486663, upper bound: 19178.4486663

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -19178.4465010, upper bound: 19178.4465010

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
time: 0.79 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 0, lower bound: -19178.4443700, upper bound: 19178.4443700

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
time: 1.13 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 0, lower bound: -19178.4443091, upper bound: 19178.4443091

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4415636, upper bound: 19178.4415636
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469
1: -1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926
2: -636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629
3: -516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145
4: -748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303

Time for backsubstitution: 2.58 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.45 + 415.73 = 420.18 seconds
