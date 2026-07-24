## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 157.33074007686602


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299)
1: (-378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955)
2: (-209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168)
3: (-347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602)
4: (-257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.97 + 2.38 = 3.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -157.3323134, upper bound: 157.3323134

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3321150, upper bound: 157.3321687
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3321150, upper bound: 157.3321150
time: 1.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 0, lower bound: -157.3321150, upper bound: 157.3321687
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 0, lower bound: -157.3321150, upper bound: 157.3321150

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
time: 0.94 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 0, lower bound: -157.3307091, upper bound: 157.3307091

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.34 + 8.11 = 11.45 seconds
