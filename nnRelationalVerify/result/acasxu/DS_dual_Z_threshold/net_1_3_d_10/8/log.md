## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.65 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6535146, upper bound: 187.6590011
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6535146
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -187.6535146, upper bound: 187.6590011
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6535146

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6396430
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6316936
time: 0.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6312887
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312887
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.94 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6396430
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6316936
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6312887
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312887

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306595, upper bound: 187.6396426
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6396430
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6310797, upper bound: 187.6316936
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6269354
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269354, upper bound: 187.6312808
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6310797
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312808
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396426, upper bound: 187.6306595
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.87 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6306595, upper bound: 187.6396426
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6396430
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6310797, upper bound: 187.6316936
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6269354
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6269354, upper bound: 187.6312808
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6310797
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312808
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.87
Output dim: 3, lower bound: -187.6396426, upper bound: 187.6306595

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280175, upper bound: 187.6392374
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302348, upper bound: 187.6329781
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6392715
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309060, upper bound: 187.6328802
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282884, upper bound: 187.6308699
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6293496
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6181620
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309475, upper bound: 187.6198169
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198169, upper bound: 187.6309475
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6181620, upper bound: 187.6286237
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293496, upper bound: 187.6308965
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308699, upper bound: 187.6282884
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6328802, upper bound: 187.6309060
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6286237
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6329781, upper bound: 187.6302348
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392374, upper bound: 187.6280175
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6280175, upper bound: 187.6392374
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6302348, upper bound: 187.6329781
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6392715
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6309060, upper bound: 187.6328802
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6282884, upper bound: 187.6308699
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6293496
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6181620
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6309475, upper bound: 187.6198169
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6198169, upper bound: 187.6309475
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6181620, upper bound: 187.6286237
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6293496, upper bound: 187.6308965
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6308699, upper bound: 187.6282884
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6328802, upper bound: 187.6309060
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6286237
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6329781, upper bound: 187.6302348
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -187.6392374, upper bound: 187.6280175

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6359014
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324970
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324059
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383040
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316535
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292397, upper bound: 187.6323326
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6298749
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290564
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289233, upper bound: 187.6169452
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289233
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300380
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298749, upper bound: 187.6279032
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6292397
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6300787
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324059, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324970, upper bound: 187.6293701
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6276140
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6359014
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324059
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383040
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316535
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6292397, upper bound: 187.6323326
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6298749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290564
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6289233, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300380
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6298749, upper bound: 187.6279032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6292397
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6300787
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6324059, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6324970, upper bound: 187.6293701
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6276140

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6359014
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6350569
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6307631
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324119
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324825
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324057
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316793
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6169452
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383040
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6316535
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316265
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6323326
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6316206
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6298642
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6297572
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290339
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6169452
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6169452
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6169452
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6287897, upper bound: 187.6169452
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6169452
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289107
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6287897
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6284335
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280248
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290339, upper bound: 187.6300380
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6279032
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6277006
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6292292
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6289546
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6300787
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6284055
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6278962
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316793, upper bound: 187.6169452
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6169452
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324825, upper bound: 187.6293701
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324119, upper bound: 187.6169452
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6169452
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6350569, upper bound: 187.6276140
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6274335
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6359014
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6350569
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6307631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324119
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324825
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316793
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6169452
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383040
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6316535
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316265
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6323326
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6316206
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6298642
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6297572
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290339
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6287897, upper bound: 187.6169452
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289107
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6287897
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6284335
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280248
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6290339, upper bound: 187.6300380
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6279032
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6277006
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6292292
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6289546
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6300787
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6284055
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6278962
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6316793, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6324825, upper bound: 187.6293701
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6324119, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6169452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6350569, upper bound: 187.6276140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.57
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6274335

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6359014
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6288339
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6350569
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6282481
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6383088
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6300629
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6307631
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6174868
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6286284
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6273223, upper bound: 187.6183649
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6290647
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6173716
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6286009
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169560
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6282866
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6209462
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6190934
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6165894
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6221780, upper bound: 187.6383040
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6290077
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6316535
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6189947
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267311, upper bound: 187.6316265
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6184694
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247816, upper bound: 187.6323326
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6189178
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252212, upper bound: 187.6316206
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6167477
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6298642
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6261777
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6297572
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6259919
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300658
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6266741
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6198769
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277836, upper bound: 187.6178818
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168195, upper bound: 187.6266759
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169055
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6165894
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6165894
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6165894
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6165894
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267553, upper bound: 187.6187512
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6165894
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6165894
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6289107
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6287897
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300938
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6187512, upper bound: 187.6267553
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6284335
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168375
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168268
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259758
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259949
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6172086, upper bound: 187.6165894
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266759, upper bound: 187.6178248
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178818, upper bound: 187.6277836
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198769, upper bound: 187.6165894
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266741, upper bound: 187.6274766
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6165894
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259919, upper bound: 187.6279032
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6165894
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261777, upper bound: 187.6277006
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6165894
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6292292
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6252212
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189178, upper bound: 187.6289546
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6247816
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6184694, upper bound: 187.6300787
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6267311
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189947, upper bound: 187.6284055
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6165894
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6321630, upper bound: 187.6171180
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6278962
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6203960, upper bound: 187.6168405
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6174273, upper bound: 187.6259671
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282866, upper bound: 187.6163102
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169560, upper bound: 187.6163102
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6165894
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290647, upper bound: 187.6178204
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6183649, upper bound: 187.6273223
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286284, upper bound: 187.6163102
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6174868, upper bound: 187.6163102
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6165894
time: 0.63 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6359014
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6288339
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6350569
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6282481
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6383088
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6300629
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6307631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6174868
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6286284
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6273223, upper bound: 187.6183649
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6290647
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6173716
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6286009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169560
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6282866
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6209462
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6190934
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6165894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6221780, upper bound: 187.6383040
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6290077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6316535
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6189947
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6267311, upper bound: 187.6316265
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6184694
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6247816, upper bound: 187.6323326
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6189178
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6252212, upper bound: 187.6316206
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6167477
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6298642
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6261777
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6297572
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6259919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300658
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6266741
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6198769
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6277836, upper bound: 187.6178818
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6168195, upper bound: 187.6266759
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169055
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6267553, upper bound: 187.6187512
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6289107
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6287897
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6187512, upper bound: 187.6267553
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6284335
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168375
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168268
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259758
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6172086, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6266759, upper bound: 187.6178248
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6178818, upper bound: 187.6277836
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6198769, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6266741, upper bound: 187.6274766
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6259919, upper bound: 187.6279032
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6165894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6261777, upper bound: 187.6277006
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6292292
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6252212
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6189178, upper bound: 187.6289546
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6247816
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6184694, upper bound: 187.6300787
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6267311
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6189947, upper bound: 187.6284055
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6321630, upper bound: 187.6171180
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6278962
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6203960, upper bound: 187.6168405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6174273, upper bound: 187.6259671
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6282866, upper bound: 187.6163102
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6169560, upper bound: 187.6163102
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6290647, upper bound: 187.6178204
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6183649, upper bound: 187.6273223
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6286284, upper bound: 187.6163102
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6174868, upper bound: 187.6163102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6165894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -187.6350569, upper bound: 187.6276140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6274335

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.30 + 417.32 = 420.62 seconds
