## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 3613.31311749156


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031)
1: (-2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305)
2: (-2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000)
3: (-1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141)
4: (-3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883)

## BASE Result
execution time: IAR + LP analysis = 1.69 + 2.03 = 3.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7618777, upper bound: 3614.7618777


# Binary Search by BASE starts (time budget: 1196.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=4653.345703125
rel_dist={0: [-3614.7615518272232, 3614.7615518272232]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=4653.345703125
rel_dist={0: [-3614.75946433006, 3614.75946433006]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=4653.345703125
rel_dist={0: [-3614.757789011659, 3614.757789011659]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=4653.345703125
rel_dist={0: [-3614.7566826429224, 3614.756682642923]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=4653.345703125
rel_dist={0: [-3614.7559892132076, 3614.7559892132085]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=4653.345703125
rel_dist={0: [-3614.7555867120263, 3614.7555867120263]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=4653.345703125
rel_dist={0: [-3614.7553802645316, 3614.7553802645307]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=4653.345703125
rel_dist={0: [-3614.7552770407865, 3614.7552770407856]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=4653.345703125
rel_dist={0: [-3614.7552254289176, 3614.7552254289185]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=4653.345703125
rel_dist={0: [-3614.7551996229913, 3614.755199622992]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=4653.345703125
rel_dist={0: [-3614.7551867114494, 3614.755186711449]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=4653.345703125
rel_dist={0: [-3614.7551802402395, 3614.7551802402395]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=4653.345703125
rel_dist={0: [-3614.755176997902, 3614.755176997902]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=4653.345703125
rel_dist={0: [-3614.7551753768576, 3614.7551753768576]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=4653.345703125
rel_dist={0: [-3614.7551745654505, 3614.755174566577]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=4653.345703125
rel_dist={0: [-3614.755174159827, 3614.755174159827]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=4653.345703125
rel_dist={0: [-3614.7551739681044, 3614.755173968104]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=4653.345703125
rel_dist={0: [-3614.755173863461, 3614.7551740795734]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=4653.345703125
rel_dist={0: [-3614.7551791817696, 3614.755176248431]}

## Binary Search Result
Binary search time: 76.57 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1119.71 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 1.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936572
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 4): status=Status.VERIFIED, low=0.9687500, high=1.0000000, mid=0.9687500, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 5) starts
Candidate diff: 0.9843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 1.05 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 5): status=Status.VERIFIED, low=0.9843750, high=1.0000000, mid=0.9843750, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 6) starts
Candidate diff: 0.9921875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.30
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 6): status=Status.VERIFIED, low=0.9921875, high=1.0000000, mid=0.9921875, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 7) starts
Candidate diff: 0.9960938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.86
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 7): status=Status.VERIFIED, low=0.9960938, high=1.0000000, mid=0.9960938, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 8) starts
Candidate diff: 0.9980469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936572
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 8): status=Status.VERIFIED, low=0.9980469, high=1.0000000, mid=0.9980469, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 9) starts
Candidate diff: 0.9990234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 9): status=Status.VERIFIED, low=0.9990234, high=1.0000000, mid=0.9990234, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 10) starts
Candidate diff: 0.9995117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 10): status=Status.VERIFIED, low=0.9995117, high=1.0000000, mid=0.9995117, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 11) starts
Candidate diff: 0.9997559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 11): status=Status.VERIFIED, low=0.9997559, high=1.0000000, mid=0.9997559, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 12) starts
Candidate diff: 0.9998779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 12): status=Status.VERIFIED, low=0.9998779, high=1.0000000, mid=0.9998779, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 13) starts
Candidate diff: 0.9999390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 13): status=Status.VERIFIED, low=0.9999390, high=1.0000000, mid=0.9999390, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 14) starts
Candidate diff: 0.9999695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737246
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 14): status=Status.VERIFIED, low=0.9999695, high=1.0000000, mid=0.9999695, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 15) starts
Candidate diff: 0.9999847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 15): status=Status.VERIFIED, low=0.9999847, high=1.0000000, mid=0.9999847, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 16) starts
Candidate diff: 0.9999924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 16): status=Status.VERIFIED, low=0.9999924, high=1.0000000, mid=0.9999924, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 17) starts
Candidate diff: 0.9999962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 17): status=Status.VERIFIED, low=0.9999962, high=1.0000000, mid=0.9999962, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 18) starts
Candidate diff: 0.9999981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 18): status=Status.VERIFIED, low=0.9999981, high=1.0000000, mid=0.9999981, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 19) starts
Candidate diff: 0.9999990


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -3614.7384831, upper bound: 3614.7435832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -3614.7435832, upper bound: 3614.7384831

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
time: 0.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737246, upper bound: 3613.6737543
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737986, upper bound: 3613.6737424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737424, upper bound: 3613.6737986
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3613.6737543, upper bound: 3613.6737246

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936400, upper bound: 3613.3936754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936139, upper bound: 3613.3936734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3937395, upper bound: 3613.3936528
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3947039, upper bound: 3613.3936572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936572, upper bound: 3613.3947039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936528, upper bound: 3613.3937395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936734, upper bound: 3613.3936139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -3613.3936754, upper bound: 3613.3936400

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031
1: -2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305
2: -2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000
3: -1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141
4: -3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915692, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915681
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1924388, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1924388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915681, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -3613.1915633, upper bound: 3613.1915633
Binary search (step 19): status=Status.VERIFIED, low=0.9999990, high=1.0000000, mid=0.9999990, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9999990463256836
execution time: 1053.86 seconds
