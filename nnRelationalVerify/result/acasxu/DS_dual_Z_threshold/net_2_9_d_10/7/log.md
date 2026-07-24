## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4711.385590072957


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836)
1: (-239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813)
2: (-164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602)
3: (-201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642)
4: (-160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.62 + 1.95 = 5.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4711.4327044, upper bound: 4711.4327043

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326788, upper bound: 4711.4326789
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326788, upper bound: 4711.4326789
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 0, lower bound: -4711.4326788, upper bound: 4711.4326789
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 0, lower bound: -4711.4326788, upper bound: 4711.4326789

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
time: 0.58 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326787
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326787
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 3.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326744, upper bound: 4711.4326745
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.70 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326744, upper bound: 4711.4326745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326744
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326744
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326744
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326745, upper bound: 4711.4326745
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836
1: -239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813
2: -164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602
3: -201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642
4: -160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925

Time for backsubstitution: 2.97 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.56 + 415.11 = 420.68 seconds
