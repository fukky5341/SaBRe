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
execution time: IAR + RelationalAnalysis = 0.77 + 2.01 = 2.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7590211, upper bound: 3614.7590211

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7391149, upper bound: 3614.7391463
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7391463, upper bound: 3614.7391149
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -3614.7391149, upper bound: 3614.7391463
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -3614.7391463, upper bound: 3614.7391149

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7390302, upper bound: 3614.7390630
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7390302, upper bound: 3614.7390320
time: 0.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7391285, upper bound: 3614.7389973
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7389960, upper bound: 3614.7391041
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.50 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -3614.7390302, upper bound: 3614.7390630
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -3614.7390302, upper bound: 3614.7390320
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -3614.7391285, upper bound: 3614.7389973
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -3614.7389960, upper bound: 3614.7391041

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3944259, upper bound: 3614.3944259
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3944259, upper bound: 3614.3944259
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7389438, upper bound: 3614.7389548
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7389438, upper bound: 3614.7390025
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7390330, upper bound: 3614.7384117
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7376370, upper bound: 3614.7389023
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7389690, upper bound: 3614.7390422
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7389659, upper bound: 3614.7390260
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.3944259, upper bound: 3614.3944259
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.3944259, upper bound: 3614.3944259
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7389438, upper bound: 3614.7389548
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7389438, upper bound: 3614.7390025
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7390330, upper bound: 3614.7384117
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7376370, upper bound: 3614.7389023
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7389690, upper bound: 3614.7390422
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -3614.7389659, upper bound: 3614.7390260

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3955161, upper bound: 3614.3929076
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3933086, upper bound: 3614.3944259
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921959, upper bound: 3614.3921959
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921959, upper bound: 3614.3937165
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7316899, upper bound: 3614.7335485
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7316899, upper bound: 3614.7310325
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7361970, upper bound: 3614.7363542
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7362953, upper bound: 3614.7363493
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7377925, upper bound: 3614.7370741
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7374622, upper bound: 3614.7361276
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185163
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7161074
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4842757, upper bound: 3614.4869524
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4868717, upper bound: 3614.4868163
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3955161, upper bound: 3614.3929076
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3933086, upper bound: 3614.3944259
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3921959, upper bound: 3614.3921959
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3921959, upper bound: 3614.3937165
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7316899, upper bound: 3614.7335485
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7316899, upper bound: 3614.7310325
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7361970, upper bound: 3614.7363542
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7362953, upper bound: 3614.7363493
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7377925, upper bound: 3614.7370741
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7374622, upper bound: 3614.7361276
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185163
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7161074
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.4842757, upper bound: 3614.4869524
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.4868717, upper bound: 3614.4868163
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3955161, upper bound: 3614.3929076
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3929076, upper bound: 3614.3929076
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3887219, upper bound: 3614.3887219
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3897206, upper bound: 3614.3887219
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3936969
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7310208, upper bound: 3614.7335426
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7310208, upper bound: 3614.7310208
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7335979, upper bound: 3614.7310325
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7335300, upper bound: 3614.7310325
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3879069, upper bound: 3613.3872808
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3879069, upper bound: 3613.3872808
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770073
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770098
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7167521
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7173800, upper bound: 3614.7147745
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7351146, upper bound: 3614.7361276
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7374508, upper bound: 3614.7357828
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185163
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185098
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6843035, upper bound: 3613.6843820
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6843035, upper bound: 3613.6843820
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4779922, upper bound: 3614.4812492
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4812865, upper bound: 3614.4797190
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811583, upper bound: 3614.2811463
time: 0.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3955161, upper bound: 3614.3929076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3929076, upper bound: 3614.3929076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3887219, upper bound: 3614.3887219
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3897206, upper bound: 3614.3887219
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3936969
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7310208, upper bound: 3614.7335426
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7310208, upper bound: 3614.7310208
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7335979, upper bound: 3614.7310325
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7335300, upper bound: 3614.7310325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3613.3879069, upper bound: 3613.3872808
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3613.3879069, upper bound: 3613.3872808
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770073
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770098
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7167521
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7173800, upper bound: 3614.7147745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7351146, upper bound: 3614.7361276
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7374508, upper bound: 3614.7357828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185163
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.7158966, upper bound: 3614.7185098
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3613.6843035, upper bound: 3613.6843820
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3613.6843035, upper bound: 3613.6843820
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.4779922, upper bound: 3614.4812492
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.4812865, upper bound: 3614.4797190
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -3614.2811583, upper bound: 3614.2811463

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3920245, upper bound: 3614.3894207
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3894207, upper bound: 3614.3894207
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3818435, upper bound: 3614.3818435
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3818435, upper bound: 3614.3818435
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2085205, upper bound: 3614.2075579
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2084282, upper bound: 3614.2075579
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3900983
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3902240
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6244602, upper bound: 3614.6269503
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6251235, upper bound: 3614.6244602
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6542369, upper bound: 3614.6542369
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6542369, upper bound: 3614.6542369
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5705113, upper bound: 3614.5680160
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5707340, upper bound: 3614.5680160
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7310264, upper bound: 3614.7310264
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7310264, upper bound: 3614.7310264
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5578231
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5556662
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5748642
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770098
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7147784, upper bound: 3614.7164272
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7147788, upper bound: 3614.7167521
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7147784, upper bound: 3614.7146224
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7147788, upper bound: 3614.7147745
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7293858, upper bound: 3614.7303868
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7293858, upper bound: 3614.7300541
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7374390, upper bound: 3614.7351146
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7351146, upper bound: 3614.7357828
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158862, upper bound: 3614.7184975
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158862, upper bound: 3614.7185045
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7170803
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172538
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6843665
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893249, upper bound: 3613.2893802
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2893249, upper bound: 3613.2893249
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5344201, upper bound: 3613.5344201
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5344201, upper bound: 3613.5344201
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3372194, upper bound: 3613.3372229
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3372296, upper bound: 3613.3372178
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4812865, upper bound: 3614.4780109
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4803748, upper bound: 3614.4797190
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7454763, upper bound: 3613.7454763
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7454763, upper bound: 3613.7454763
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2740141, upper bound: 3614.2740141
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2740141, upper bound: 3614.2740141
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.80 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3920245, upper bound: 3614.3894207
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3894207, upper bound: 3614.3894207
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3928996, upper bound: 3614.3928996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3818435, upper bound: 3614.3818435
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3818435, upper bound: 3614.3818435
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2085205, upper bound: 3614.2075579
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2084282, upper bound: 3614.2075579
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3921659, upper bound: 3614.3921659
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3900983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3902240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.6244602, upper bound: 3614.6269503
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.6251235, upper bound: 3614.6244602
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.6542369, upper bound: 3614.6542369
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.6542369, upper bound: 3614.6542369
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5705113, upper bound: 3614.5680160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5707340, upper bound: 3614.5680160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7310264, upper bound: 3614.7310264
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7310264, upper bound: 3614.7310264
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5578231
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5556662
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5748642
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.5747658, upper bound: 3614.5770098
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7147784, upper bound: 3614.7164272
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7147788, upper bound: 3614.7167521
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7147784, upper bound: 3614.7146224
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7147788, upper bound: 3614.7147745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7293858, upper bound: 3614.7303868
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7293858, upper bound: 3614.7300541
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7374390, upper bound: 3614.7351146
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7351146, upper bound: 3614.7357828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7158862, upper bound: 3614.7184975
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7158862, upper bound: 3614.7185045
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7170803
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172538
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6843665
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.2893249, upper bound: 3613.2893802
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.2893249, upper bound: 3613.2893249
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.1319637, upper bound: 3614.1319637
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.5344201, upper bound: 3613.5344201
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.5344201, upper bound: 3613.5344201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3372194, upper bound: 3613.3372229
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.3372296, upper bound: 3613.3372178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.4812865, upper bound: 3614.4780109
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.4803748, upper bound: 3614.4797190
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.7454763, upper bound: 3613.7454763
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3613.7454763, upper bound: 3613.7454763
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2740141, upper bound: 3614.2740141
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2740141, upper bound: 3614.2740141
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045565, upper bound: 3614.3045565
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045565, upper bound: 3614.3045565
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3402215, upper bound: 3613.3402215
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3402215, upper bound: 3613.3402215
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3894126, upper bound: 3614.3894126
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3894126, upper bound: 3614.3894126
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3330014, upper bound: 3613.3330014
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3330014, upper bound: 3613.3330014
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707518
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707398
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075579, upper bound: 3614.2075579
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075579, upper bound: 3614.2075579
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2033157, upper bound: 3614.2033157
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2041737, upper bound: 3614.2033157
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3833708, upper bound: 3614.3833708
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3833708, upper bound: 3614.3833708
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3809854, upper bound: 3614.3809854
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3813164, upper bound: 3614.3809854
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3886921
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3900983
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079061
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079051
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6034486, upper bound: 3614.6058969
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6057375, upper bound: 3614.6032476
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6231058, upper bound: 3614.6231058
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6244007, upper bound: 3614.6231058
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6562610, upper bound: 3614.6539664
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6539664, upper bound: 3614.6539664
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6511059, upper bound: 3614.6511059
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6511059, upper bound: 3614.6511059
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4829495, upper bound: 3614.4805635
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4805635, upper bound: 3614.4805635
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5680160, upper bound: 3614.5680160
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5680160, upper bound: 3614.5680160
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6244648, upper bound: 3614.6244648
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6261910, upper bound: 3614.6244648
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7309594, upper bound: 3614.7309594
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7309594, upper bound: 3614.7309594
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3864433, upper bound: 3613.3863396
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3863396, upper bound: 3613.3863396
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3827572, upper bound: 3613.3827447
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3827936, upper bound: 3613.3827447
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3820256, upper bound: 3613.3820256
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3820256, upper bound: 3613.3820256
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2713319, upper bound: 3614.2733526
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2713319, upper bound: 3614.2733526
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2416361, upper bound: 3613.2416102
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2416361, upper bound: 3613.2416102
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5746753, upper bound: 3614.5746753
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5746753, upper bound: 3614.5746753
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5578304
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5556389
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3496013, upper bound: 3614.3496013
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3496013, upper bound: 3614.3496013
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591791
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591790, upper bound: 3613.5591791
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591784
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7313298, upper bound: 3614.7293848
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7293848, upper bound: 3614.7303868
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7253889, upper bound: 3614.7253889
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7253889, upper bound: 3614.7260861
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146110, upper bound: 3614.7146110
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7170387, upper bound: 3614.7146110
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7350879, upper bound: 3614.7328296
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7337238, upper bound: 3614.7335255
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158412, upper bound: 3614.7184784
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7158412, upper bound: 3614.7184834
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4657022, upper bound: 3614.4693387
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4657022, upper bound: 3614.4693458
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7114630, upper bound: 3614.7138149
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7114630, upper bound: 3614.7138273
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172538
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172493
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6843665
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1235272, upper bound: 3614.1235272
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1235272, upper bound: 3614.1235272
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1298332, upper bound: 3614.1298332
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1298332, upper bound: 3614.1298332
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5318390, upper bound: 3613.5318390
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5318390, upper bound: 3613.5318390
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4115118, upper bound: 3613.4115118
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4115118, upper bound: 3613.4115118
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3262788, upper bound: 3613.3262855
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3262475, upper bound: 3613.3262881
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0455298, upper bound: 3613.0455298
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0465367, upper bound: 3613.0456060
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4775019, upper bound: 3614.4742140
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4742004, upper bound: 3614.4742186
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3724317, upper bound: 3614.3732166
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3724317, upper bound: 3614.3745572
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6223857, upper bound: 3613.6223857
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6223857, upper bound: 3613.6223857
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7326983, upper bound: 3613.7326983
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7326983, upper bound: 3613.7326983
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5648350, upper bound: 3613.5648350
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5648350, upper bound: 3613.5648350
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0490143, upper bound: 3614.0490143
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0490143, upper bound: 3614.0490143
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
time: 0.89 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3045565, upper bound: 3614.3045565
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3045565, upper bound: 3614.3045565
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3402215, upper bound: 3613.3402215
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3402215, upper bound: 3613.3402215
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811463, upper bound: 3614.2811463
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3894126, upper bound: 3614.3894126
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3894126, upper bound: 3614.3894126
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3330014, upper bound: 3613.3330014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3330014, upper bound: 3613.3330014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707518
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707398
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2075579, upper bound: 3614.2075579
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2075579, upper bound: 3614.2075579
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2033157, upper bound: 3614.2033157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2041737, upper bound: 3614.2033157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3887159, upper bound: 3614.3887159
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3833708, upper bound: 3614.3833708
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3833708, upper bound: 3614.3833708
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3809854, upper bound: 3614.3809854
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3813164, upper bound: 3614.3809854
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3886921
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3886921, upper bound: 3614.3900983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079061
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079051
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6034486, upper bound: 3614.6058969
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6057375, upper bound: 3614.6032476
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6231058, upper bound: 3614.6231058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6244007, upper bound: 3614.6231058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6562610, upper bound: 3614.6539664
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6539664, upper bound: 3614.6539664
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6511059, upper bound: 3614.6511059
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6511059, upper bound: 3614.6511059
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4829495, upper bound: 3614.4805635
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4805635, upper bound: 3614.4805635
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5680160, upper bound: 3614.5680160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5680160, upper bound: 3614.5680160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6244648, upper bound: 3614.6244648
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.6261910, upper bound: 3614.6244648
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7309594, upper bound: 3614.7309594
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7309594, upper bound: 3614.7309594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3864433, upper bound: 3613.3863396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3863396, upper bound: 3613.3863396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3827572, upper bound: 3613.3827447
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3827936, upper bound: 3613.3827447
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3820256, upper bound: 3613.3820256
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3820256, upper bound: 3613.3820256
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3864764, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3863726, upper bound: 3613.3863726
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2713319, upper bound: 3614.2733526
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2713319, upper bound: 3614.2733526
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.2416361, upper bound: 3613.2416102
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.2416361, upper bound: 3613.2416102
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5746753, upper bound: 3614.5746753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5746753, upper bound: 3614.5746753
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5578304
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.5556389, upper bound: 3614.5556389
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3496013, upper bound: 3614.3496013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3496013, upper bound: 3614.3496013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591791
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591790, upper bound: 3613.5591791
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591784
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5591789, upper bound: 3613.5591789
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7313298, upper bound: 3614.7293848
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7293848, upper bound: 3614.7303868
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7253889, upper bound: 3614.7253889
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7253889, upper bound: 3614.7260861
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7146110, upper bound: 3614.7146110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7170387, upper bound: 3614.7146110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7350879, upper bound: 3614.7328296
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7337238, upper bound: 3614.7335255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7158412, upper bound: 3614.7184784
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7158412, upper bound: 3614.7184834
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4657022, upper bound: 3614.4693387
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4657022, upper bound: 3614.4693458
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7114630, upper bound: 3614.7138149
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7114630, upper bound: 3614.7138273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.7146224, upper bound: 3614.7172493
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6842899
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6842899, upper bound: 3613.6843665
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.1235272, upper bound: 3614.1235272
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.1235272, upper bound: 3614.1235272
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.1298332, upper bound: 3614.1298332
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.1298332, upper bound: 3614.1298332
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5318390, upper bound: 3613.5318390
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5318390, upper bound: 3613.5318390
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.4115118, upper bound: 3613.4115118
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.4115118, upper bound: 3613.4115118
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3262788, upper bound: 3613.3262855
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.3262475, upper bound: 3613.3262881
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.0455298, upper bound: 3613.0455298
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.0465367, upper bound: 3613.0456060
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4775019, upper bound: 3614.4742140
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.4742004, upper bound: 3614.4742186
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3724317, upper bound: 3614.3732166
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.3724317, upper bound: 3614.3745572
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6223857, upper bound: 3613.6223857
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.6223857, upper bound: 3613.6223857
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.7326983, upper bound: 3613.7326983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.7326983, upper bound: 3613.7326983
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5648350, upper bound: 3613.5648350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3613.5648350, upper bound: 3613.5648350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.0490143, upper bound: 3614.0490143
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.0490143, upper bound: 3614.0490143
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.77
Output dim: 0, lower bound: -3614.2811311, upper bound: 3614.2811311

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1879265, upper bound: 3614.1879265
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1902793, upper bound: 3614.1879265
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045498, upper bound: 3614.3045498
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045498, upper bound: 3614.3045498
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3402113, upper bound: 3613.3402113
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3402113, upper bound: 3613.3402113
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2594015, upper bound: 3613.2594015
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2594015, upper bound: 3613.2594015
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2764034, upper bound: 3614.2764034
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2764034, upper bound: 3614.2764034
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2764034, upper bound: 3614.2764034
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2764034, upper bound: 3614.2764034
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2760933, upper bound: 3614.2760933
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2760933, upper bound: 3614.2760933
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1910765, upper bound: 3614.1910765
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1910765, upper bound: 3614.1910765
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5941510, upper bound: 3613.5941510
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5941510, upper bound: 3613.5941510
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3875979, upper bound: 3614.3875979
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045498, upper bound: 3614.3045498
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3045498, upper bound: 3614.3045498
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2779824, upper bound: 3614.2779824
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2779824, upper bound: 3614.2779824
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3276504, upper bound: 3613.3276504
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3276504, upper bound: 3613.3276504
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3164917, upper bound: 3613.3164917
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3164917, upper bound: 3613.3164917
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707518
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2707398, upper bound: 3614.2707398
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1323534, upper bound: 3614.1323534
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1323534, upper bound: 3614.1323534
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2037980, upper bound: 3614.2028738
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2028738, upper bound: 3614.2028738
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1982770, upper bound: 3614.1982770
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1982770, upper bound: 3614.1982770
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2033157, upper bound: 3614.2033157
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2033157, upper bound: 3614.2033157
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2033150, upper bound: 3614.2033150
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2041737, upper bound: 3614.2033150
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3774923, upper bound: 3614.3774923
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3774923, upper bound: 3614.3774923
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2779673, upper bound: 3614.2779673
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2779673, upper bound: 3614.2779673
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3833419, upper bound: 3614.3833419
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3833419, upper bound: 3614.3833419
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7308260, upper bound: 3613.7308260
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7308260, upper bound: 3613.7308260
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3075997, upper bound: 3614.3075997
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2021800, upper bound: 3614.2021800
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2021800, upper bound: 3614.2021800
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3777870, upper bound: 3614.3774686
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3774686, upper bound: 3614.3774686
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3048510, upper bound: 3614.3040387
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3047918, upper bound: 3614.3040387
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079470
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2075561, upper bound: 3614.2079470
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1615624, upper bound: 3614.1615624
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1615624, upper bound: 3614.1615624
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0374390, upper bound: 3614.0381248
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0374390, upper bound: 3614.0374390
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6271093, upper bound: 3613.6271093
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6271093, upper bound: 3613.6271093
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6032429, upper bound: 3614.6032429
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6032429, upper bound: 3614.6032429
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3942938, upper bound: 3614.3942938
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3960166, upper bound: 3614.3942938
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6188907, upper bound: 3614.6188907
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6188907, upper bound: 3614.6188907
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6562610, upper bound: 3614.6539664
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6539664, upper bound: 3614.6539664
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6503301, upper bound: 3614.6503301
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6503301, upper bound: 3614.6503301
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6324002, upper bound: 3614.6324002
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.6324002, upper bound: 3614.6324002
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.14 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.79 + 417.55 = 420.34 seconds
