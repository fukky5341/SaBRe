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
execution time: IAR + RelationalAnalysis = 1.03 + 2.40 = 3.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -157.3323134, upper bound: 157.3323134

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3322890, upper bound: 157.3322890
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3322890, upper bound: 157.3323050
time: 0.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.99 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 0, lower bound: -157.3322890, upper bound: 157.3322890
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 0, lower bound: -157.3322890, upper bound: 157.3323050

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3322866, upper bound: 157.3322873
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3322866, upper bound: 157.3322866
time: 0.96 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3290536, upper bound: 157.3290602
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3290536, upper bound: 157.3290602
time: 1.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.07 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -157.3322866, upper bound: 157.3322873
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -157.3322866, upper bound: 157.3322866
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.07
Output dim: 0, lower bound: -157.3290536, upper bound: 157.3290602
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.07
Output dim: 0, lower bound: -157.3290536, upper bound: 157.3290602

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3293195, upper bound: 157.3293195
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3293195, upper bound: 157.3293195
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3317042, upper bound: 157.3317000
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3316840, upper bound: 157.3317000
time: 0.95 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -157.3293195, upper bound: 157.3293195
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -157.3293195, upper bound: 157.3293195
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -157.3317042, upper bound: 157.3317000
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -157.3316840, upper bound: 157.3317000

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3316411, upper bound: 157.3316411
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3316572, upper bound: 157.3316465
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3316693, upper bound: 157.3316693
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3316693, upper bound: 157.3316693
time: 0.96 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.94 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -157.3316411, upper bound: 157.3316411
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -157.3316572, upper bound: 157.3316465
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -157.3316693, upper bound: 157.3316693
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -157.3316693, upper bound: 157.3316693

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307238, upper bound: 157.3307212
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307212, upper bound: 157.3307212
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307239, upper bound: 157.3307212
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307212, upper bound: 157.3307212
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309558, upper bound: 157.3309684
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309558, upper bound: 157.3309684
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304717, upper bound: 157.3304717
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304717, upper bound: 157.3304717
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.15 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3307238, upper bound: 157.3307212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3307212, upper bound: 157.3307212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3307239, upper bound: 157.3307212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3307212, upper bound: 157.3307212
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3309558, upper bound: 157.3309684
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3309558, upper bound: 157.3309684
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3304717, upper bound: 157.3304717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.15
Output dim: 0, lower bound: -157.3304717, upper bound: 157.3304717

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306362, upper bound: 157.3306327
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306327, upper bound: 157.3306327
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309664
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309546
time: 1.00 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.14 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 0, lower bound: -157.3306362, upper bound: 157.3306327
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 0, lower bound: -157.3306327, upper bound: 157.3306327
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309664
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309546

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309655
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309664
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
time: 1.07 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.87 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309655
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -157.3309546, upper bound: 157.3309664
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309277, upper bound: 157.3309277
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309277, upper bound: 157.3309277
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
time: 1.09 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3305495, upper bound: 157.3305495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3309277, upper bound: 157.3309277
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3309277, upper bound: 157.3309277
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.72
Output dim: 0, lower bound: -157.3309298, upper bound: 157.3309298

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306326, upper bound: 157.3306326
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306326, upper bound: 157.3306326
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309247, upper bound: 157.3309247
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309247, upper bound: 157.3309247
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309039, upper bound: 157.3309039
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3309039, upper bound: 157.3309039
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304193, upper bound: 157.3304193
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304193, upper bound: 157.3304193
time: 1.02 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3306326, upper bound: 157.3306326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3306326, upper bound: 157.3306326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3309247, upper bound: 157.3309247
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3309247, upper bound: 157.3309247
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3309039, upper bound: 157.3309039
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3309039, upper bound: 157.3309039
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3304193, upper bound: 157.3304193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.24
Output dim: 0, lower bound: -157.3304193, upper bound: 157.3304193

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3299115, upper bound: 157.3299115
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3299115, upper bound: 157.3299115
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308994, upper bound: 157.3308994
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308994, upper bound: 157.3308994
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308866, upper bound: 157.3308866
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308866, upper bound: 157.3308866
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308880, upper bound: 157.3308880
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308880, upper bound: 157.3308880
time: 0.92 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.22 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3299115, upper bound: 157.3299115
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3299115, upper bound: 157.3299115
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308994, upper bound: 157.3308994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308994, upper bound: 157.3308994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308866, upper bound: 157.3308866
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308866, upper bound: 157.3308866
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308880, upper bound: 157.3308880
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.22
Output dim: 0, lower bound: -157.3308880, upper bound: 157.3308880

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305522, upper bound: 157.3305522
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305522, upper bound: 157.3305522
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3289103, upper bound: 157.3289103
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3289103, upper bound: 157.3289103
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305715, upper bound: 157.3305715
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305715, upper bound: 157.3305715
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306811, upper bound: 157.3306811
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306811, upper bound: 157.3306811
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3300913, upper bound: 157.3300913
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3300913, upper bound: 157.3300913
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
time: 0.97 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 3.29 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3305522, upper bound: 157.3305522
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3305522, upper bound: 157.3305522
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3289103, upper bound: 157.3289103
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3289103, upper bound: 157.3289103
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3305715, upper bound: 157.3305715
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3305715, upper bound: 157.3305715
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3306811, upper bound: 157.3306811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3306811, upper bound: 157.3306811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3300913, upper bound: 157.3300913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3300913, upper bound: 157.3300913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 3.29
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
time: 0.99 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 2.95 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 12, time: 2.95
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 12, time: 2.95
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 12, time: 2.95
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 12, time: 2.95
Output dim: 0, lower bound: -157.3308832, upper bound: 157.3308832

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308561, upper bound: 157.3308561
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308561, upper bound: 157.3308561
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3303718, upper bound: 157.3303718
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3303718, upper bound: 157.3303718
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3296196, upper bound: 157.3296196
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3296196, upper bound: 157.3296196
time: 0.93 seconds

## Summary of splitting (split count: 12)
- Time for DS candidates: 2.82 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3308561, upper bound: 157.3308561
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3308561, upper bound: 157.3308561
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3308795, upper bound: 157.3308795
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3303718, upper bound: 157.3303718
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3303718, upper bound: 157.3303718
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3296196, upper bound: 157.3296196
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 13, time: 2.82
Output dim: 0, lower bound: -157.3296196, upper bound: 157.3296196

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304618, upper bound: 157.3304618
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304618, upper bound: 157.3304618
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3303479, upper bound: 157.3303479
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3303479, upper bound: 157.3303479
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305650, upper bound: 157.3305650
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305650, upper bound: 157.3305650
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299
1: -378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955
2: -209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168
3: -347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602
4: -257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306442, upper bound: 157.3306442
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3306442, upper bound: 157.3306442
time: 0.87 seconds

## Summary of splitting (split count: 13)
- Time for DS candidates: 3.57 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3304618, upper bound: 157.3304618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3304618, upper bound: 157.3304618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3303479, upper bound: 157.3303479
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3303479, upper bound: 157.3303479
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3305650, upper bound: 157.3305650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3305650, upper bound: 157.3305650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3306442, upper bound: 157.3306442
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 14, time: 3.57
Output dim: 0, lower bound: -157.3306442, upper bound: 157.3306442

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.43 + 133.47 = 136.90 seconds
