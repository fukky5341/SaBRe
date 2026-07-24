## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 3613.31311749156


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031)
1: (-2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305)
2: (-2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000)
3: (-1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141)
4: (-3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 2.06 = 3.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7590211, upper bound: 3614.7590211

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7381187, upper bound: 3614.7406664
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7381187, upper bound: 3614.7381187
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -3614.7381187, upper bound: 3614.7406664
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -3614.7381187, upper bound: 3614.7381187

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6736228, upper bound: 3613.6736517
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6736835, upper bound: 3613.6736495
time: 0.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6736495, upper bound: 3613.6736835
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6736517, upper bound: 3613.6736228
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.32 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6736228, upper bound: 3613.6736517
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6736835, upper bound: 3613.6736495
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6736495, upper bound: 3613.6736835
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6736517, upper bound: 3613.6736228

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3935492, upper bound: 3613.3936292
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3935394, upper bound: 3613.3936318
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936691, upper bound: 3613.3936101
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946407, upper bound: 3613.3936210
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936210, upper bound: 3613.3946407
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936101, upper bound: 3613.3936691
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936318, upper bound: 3613.3935394
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936292, upper bound: 3613.3935492
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.50 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3935492, upper bound: 3613.3936292
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3935394, upper bound: 3613.3936318
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3936691, upper bound: 3613.3936101
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3946407, upper bound: 3613.3936210
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3936210, upper bound: 3613.3946407
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3936101, upper bound: 3613.3936691
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3936318, upper bound: 3613.3935394
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.3936292, upper bound: 3613.3935492

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894263, upper bound: 3613.2893875
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893881, upper bound: 3613.2894352
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894263, upper bound: 3613.2894720
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893881, upper bound: 3613.2894815
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2895650, upper bound: 3613.2893875
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2895609, upper bound: 3613.2894283
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2903465, upper bound: 3613.2894723
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2903464, upper bound: 3613.2894823
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894283, upper bound: 3613.2903464
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2903465
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894283, upper bound: 3613.2895609
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2895650
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894815, upper bound: 3613.2893881
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2894720, upper bound: 3613.2894263
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2894725
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2894827
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894263, upper bound: 3613.2893875
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893881, upper bound: 3613.2894352
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894263, upper bound: 3613.2894720
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893881, upper bound: 3613.2894815
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2895650, upper bound: 3613.2893875
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2895609, upper bound: 3613.2894283
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2903465, upper bound: 3613.2894723
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2903464, upper bound: 3613.2894823
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894283, upper bound: 3613.2903464
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2903465
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894283, upper bound: 3613.2895609
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2895650
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894815, upper bound: 3613.2893881
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2894720, upper bound: 3613.2894263
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2894725
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -3613.2893875, upper bound: 3613.2894827

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.75 + 48.08 = 51.82 seconds
