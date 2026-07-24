## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1339.202386442214


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046)
1: (-994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672)
2: (-529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268)
3: (-910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723)
4: (-664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.97 + 2.36 = 3.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1339.2157786, upper bound: 1339.2157786

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2152724, upper bound: 1339.2155915
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2155915, upper bound: 1339.2152724
time: 1.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 3, lower bound: -1339.2152724, upper bound: 1339.2155915
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 3, lower bound: -1339.2155915, upper bound: 1339.2152724

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2148953, upper bound: 1339.2150241
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2148953, upper bound: 1339.2152071
time: 1.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2142158, upper bound: 1339.2142593
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2146458, upper bound: 1339.2142158
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -1339.2148953, upper bound: 1339.2150241
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -1339.2148953, upper bound: 1339.2152071
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -1339.2142158, upper bound: 1339.2142593
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -1339.2146458, upper bound: 1339.2142158

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2067014, upper bound: 1339.2067012
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2067012, upper bound: 1339.2067014
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141641, upper bound: 1339.2145901
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2142972, upper bound: 1339.2144449
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144188, upper bound: 1339.2141540
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141025, upper bound: 1339.2141670
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2135991, upper bound: 1339.2136680
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2140465, upper bound: 1339.2136680
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.94 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2067014, upper bound: 1339.2067012
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2067012, upper bound: 1339.2067014
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2141641, upper bound: 1339.2145901
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2142972, upper bound: 1339.2144449
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2144188, upper bound: 1339.2141540
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2141025, upper bound: 1339.2141670
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2135991, upper bound: 1339.2136680
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1339.2140465, upper bound: 1339.2136680

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2008757, upper bound: 1339.2008253
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2009353, upper bound: 1339.2008276
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2066895, upper bound: 1339.2064625
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2065030, upper bound: 1339.2066898
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141640, upper bound: 1339.2145810
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141641, upper bound: 1339.2144846
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2136487, upper bound: 1339.2137843
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2135197, upper bound: 1339.2136809
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141783, upper bound: 1339.2141540
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144188, upper bound: 1339.2140533
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2136713, upper bound: 1339.2136576
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2136713, upper bound: 1339.2136297
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2135439, upper bound: 1339.2135286
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2139475, upper bound: 1339.2136434
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102709, upper bound: 1339.2102074
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102783, upper bound: 1339.2101449
time: 0.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2008757, upper bound: 1339.2008253
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2009353, upper bound: 1339.2008276
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2066895, upper bound: 1339.2064625
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2065030, upper bound: 1339.2066898
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2141640, upper bound: 1339.2145810
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2141641, upper bound: 1339.2144846
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2136487, upper bound: 1339.2137843
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2135197, upper bound: 1339.2136809
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2141783, upper bound: 1339.2141540
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2144188, upper bound: 1339.2140533
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2136713, upper bound: 1339.2136576
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2136713, upper bound: 1339.2136297
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2135439, upper bound: 1339.2135286
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2139475, upper bound: 1339.2136434
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2102709, upper bound: 1339.2102074
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 3, lower bound: -1339.2102783, upper bound: 1339.2101449

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1981708, upper bound: 1339.1981385
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1982954, upper bound: 1339.1981263
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1685418, upper bound: 1339.1686231
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1685418, upper bound: 1339.1686231
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2139587, upper bound: 1339.2142577
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138921, upper bound: 1339.2142479
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110624, upper bound: 1339.2113375
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110693, upper bound: 1339.2114213
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090158, upper bound: 1339.2092895
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090127, upper bound: 1339.2094803
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2120374, upper bound: 1339.2122049
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2119257, upper bound: 1339.2122049
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2140562, upper bound: 1339.2141540
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141783, upper bound: 1339.2140312
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132637, upper bound: 1339.2130434
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133932, upper bound: 1339.2130017
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124643, upper bound: 1339.2126593
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124643, upper bound: 1339.2126280
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129604, upper bound: 1339.2124994
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124025, upper bound: 1339.2124386
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101390, upper bound: 1339.2099283
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101390, upper bound: 1339.2099323
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133343, upper bound: 1339.2130098
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130098, upper bound: 1339.2130098
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101744, upper bound: 1339.2101071
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101771, upper bound: 1339.2101007
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085439, upper bound: 1339.2086956
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085439, upper bound: 1339.2086966
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.92 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.1981708, upper bound: 1339.1981385
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.1982954, upper bound: 1339.1981263
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.1685418, upper bound: 1339.1686231
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.1685418, upper bound: 1339.1686231
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2139587, upper bound: 1339.2142577
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2138921, upper bound: 1339.2142479
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2110624, upper bound: 1339.2113375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2110693, upper bound: 1339.2114213
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2090158, upper bound: 1339.2092895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2090127, upper bound: 1339.2094803
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2120374, upper bound: 1339.2122049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2119257, upper bound: 1339.2122049
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2140562, upper bound: 1339.2141540
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2141783, upper bound: 1339.2140312
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2132637, upper bound: 1339.2130434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2133932, upper bound: 1339.2130017
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2124643, upper bound: 1339.2126593
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2124643, upper bound: 1339.2126280
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2129604, upper bound: 1339.2124994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2124025, upper bound: 1339.2124386
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2101390, upper bound: 1339.2099283
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2101390, upper bound: 1339.2099323
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2133343, upper bound: 1339.2130098
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2130098, upper bound: 1339.2130098
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2101744, upper bound: 1339.2101071
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2101771, upper bound: 1339.2101007
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2085439, upper bound: 1339.2086956
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 3, lower bound: -1339.2085439, upper bound: 1339.2086966

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2087367, upper bound: 1339.2091268
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086701, upper bound: 1339.2091268
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2092918, upper bound: 1339.2096012
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2092918, upper bound: 1339.2095294
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110245, upper bound: 1339.2110927
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2109799, upper bound: 1339.2112527
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107838, upper bound: 1339.2111961
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107838, upper bound: 1339.2110145
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085159, upper bound: 1339.2085575
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085163, upper bound: 1339.2088682
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090323, upper bound: 1339.2092306
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089411, upper bound: 1339.2094469
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2095002, upper bound: 1339.2095293
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096192, upper bound: 1339.2096880
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083349, upper bound: 1339.2086706
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083349, upper bound: 1339.2083349
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2135335, upper bound: 1339.2136563
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2136641
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141489, upper bound: 1339.2140312
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2141782, upper bound: 1339.2138205
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080045, upper bound: 1339.2081737
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081543, upper bound: 1339.2082034
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132258, upper bound: 1339.2127623
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127556, upper bound: 1339.2127556
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2105571, upper bound: 1339.2107939
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2105571, upper bound: 1339.2107090
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112450
time: 11.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112477, upper bound: 1339.2113689
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2122277, upper bound: 1339.2122287
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2122277, upper bound: 1339.2122844
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086526, upper bound: 1339.2084576
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089673, upper bound: 1339.2083075
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086593, upper bound: 1339.2085164
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085188, upper bound: 1339.2085164
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2066858, upper bound: 1339.2066759
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2067029, upper bound: 1339.2066759
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133343, upper bound: 1339.2130098
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130098, upper bound: 1339.2130098
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2023137, upper bound: 1339.2023137
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2023621, upper bound: 1339.2023137
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099224, upper bound: 1339.2099526
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100634, upper bound: 1339.2100833
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100778, upper bound: 1339.2099800
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100730, upper bound: 1339.2099743
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045939, upper bound: 1339.2046195
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045920, upper bound: 1339.2046121
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080928, upper bound: 1339.2080928
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084817, upper bound: 1339.2081263
time: 1.04 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.39 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2087367, upper bound: 1339.2091268
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2086701, upper bound: 1339.2091268
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2092918, upper bound: 1339.2096012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2092918, upper bound: 1339.2095294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2110245, upper bound: 1339.2110927
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2109799, upper bound: 1339.2112527
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2107838, upper bound: 1339.2111961
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2107838, upper bound: 1339.2110145
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2085159, upper bound: 1339.2085575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2085163, upper bound: 1339.2088682
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2090323, upper bound: 1339.2092306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2089411, upper bound: 1339.2094469
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2095002, upper bound: 1339.2095293
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2096192, upper bound: 1339.2096880
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2083349, upper bound: 1339.2086706
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2083349, upper bound: 1339.2083349
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2135335, upper bound: 1339.2136563
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2136641
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2141489, upper bound: 1339.2140312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2141782, upper bound: 1339.2138205
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2080045, upper bound: 1339.2081737
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2081543, upper bound: 1339.2082034
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2132258, upper bound: 1339.2127623
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2127556, upper bound: 1339.2127556
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2105571, upper bound: 1339.2107939
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2105571, upper bound: 1339.2107090
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2112477, upper bound: 1339.2113689
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2122277, upper bound: 1339.2122287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2122277, upper bound: 1339.2122844
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2086526, upper bound: 1339.2084576
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2089673, upper bound: 1339.2083075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2086593, upper bound: 1339.2085164
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2085188, upper bound: 1339.2085164
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2066858, upper bound: 1339.2066759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2067029, upper bound: 1339.2066759
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2133343, upper bound: 1339.2130098
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2130098, upper bound: 1339.2130098
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2023137, upper bound: 1339.2023137
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2023621, upper bound: 1339.2023137
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2099224, upper bound: 1339.2099526
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2100634, upper bound: 1339.2100833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2100778, upper bound: 1339.2099800
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2100730, upper bound: 1339.2099743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2045939, upper bound: 1339.2046195
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2045920, upper bound: 1339.2046121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2080928, upper bound: 1339.2080928
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 3, lower bound: -1339.2084817, upper bound: 1339.2081263

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086209, upper bound: 1339.2090027
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085604, upper bound: 1339.2089052
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086602, upper bound: 1339.2091045
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086602, upper bound: 1339.2090512
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2088384
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083921, upper bound: 1339.2085533
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083921, upper bound: 1339.2083921
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110211, upper bound: 1339.2110918
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110231, upper bound: 1339.2109655
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096366, upper bound: 1339.2098528
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096366, upper bound: 1339.2098528
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2092680, upper bound: 1339.2096062
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2092680, upper bound: 1339.2095648
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2103040, upper bound: 1339.2104862
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2103028, upper bound: 1339.2102276
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079824, upper bound: 1339.2079824
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079824, upper bound: 1339.2079824
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2024832, upper bound: 1339.2026594
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2024832, upper bound: 1339.2026594
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2076014, upper bound: 1339.2076943
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2076014, upper bound: 1339.2076943
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088595, upper bound: 1339.2094176
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089107, upper bound: 1339.2092860
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074796, upper bound: 1339.2074647
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074851, upper bound: 1339.2074647
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083892, upper bound: 1339.2087692
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083892, upper bound: 1339.2085365
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086332
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086159
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2135731
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2136535
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132980, upper bound: 1339.2135805
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132979, upper bound: 1339.2136638
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137122, upper bound: 1339.2136575
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137555, upper bound: 1339.2134516
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2079419
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2081356
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2022741, upper bound: 1339.2022741
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2022741, upper bound: 1339.2022741
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2113487
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2114700
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2104315
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2100379
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2033820
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2034120
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112450
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112437
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112460
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112477, upper bound: 1339.2113689
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2082403
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2081402
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2115141, upper bound: 1339.2114353
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2117688, upper bound: 1339.2114313
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089250, upper bound: 1339.2083075
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072714, upper bound: 1339.2072714
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073047, upper bound: 1339.2072714
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084957, upper bound: 1339.2084957
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085416, upper bound: 1339.2084957
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2065138, upper bound: 1339.2064924
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2065007, upper bound: 1339.2064924
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2063318, upper bound: 1339.2061163
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2064164, upper bound: 1339.2061142
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1942914, upper bound: 1339.1942914
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1942914, upper bound: 1339.1942914
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091634, upper bound: 1339.2093296
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091391, upper bound: 1339.2093041
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1921156, upper bound: 1339.1920614
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1921151, upper bound: 1339.1920614
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044955
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044885
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084435, upper bound: 1339.2080436
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080436, upper bound: 1339.2080924
time: 1.11 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.40 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2086209, upper bound: 1339.2090027
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2085604, upper bound: 1339.2089052
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2086602, upper bound: 1339.2091045
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2086602, upper bound: 1339.2090512
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2088384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083921, upper bound: 1339.2085533
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083921, upper bound: 1339.2083921
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2110211, upper bound: 1339.2110918
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2110231, upper bound: 1339.2109655
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2096366, upper bound: 1339.2098528
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2096366, upper bound: 1339.2098528
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2092680, upper bound: 1339.2096062
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2092680, upper bound: 1339.2095648
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2103040, upper bound: 1339.2104862
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2103028, upper bound: 1339.2102276
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2079824, upper bound: 1339.2079824
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2079824, upper bound: 1339.2079824
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2024832, upper bound: 1339.2026594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2024832, upper bound: 1339.2026594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2076014, upper bound: 1339.2076943
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2076014, upper bound: 1339.2076943
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2088595, upper bound: 1339.2094176
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2089107, upper bound: 1339.2092860
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2074796, upper bound: 1339.2074647
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2074851, upper bound: 1339.2074647
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083892, upper bound: 1339.2087692
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083892, upper bound: 1339.2085365
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086332
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086159
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2135731
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2136535
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2132980, upper bound: 1339.2135805
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2132979, upper bound: 1339.2136638
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2137122, upper bound: 1339.2136575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2137555, upper bound: 1339.2134516
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2079419
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2081356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2022741, upper bound: 1339.2022741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2022741, upper bound: 1339.2022741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2113487
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2114700
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2104315
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2100379
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2033820
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2034120
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112437
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2112477, upper bound: 1339.2113689
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2082403
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2081402
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2115141, upper bound: 1339.2114353
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2117688, upper bound: 1339.2114313
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2089250, upper bound: 1339.2083075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2072714, upper bound: 1339.2072714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2073047, upper bound: 1339.2072714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2084957, upper bound: 1339.2084957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2085416, upper bound: 1339.2084957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2065138, upper bound: 1339.2064924
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2065007, upper bound: 1339.2064924
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2063318, upper bound: 1339.2061163
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2064164, upper bound: 1339.2061142
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.1942914, upper bound: 1339.1942914
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.1942914, upper bound: 1339.1942914
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2091634, upper bound: 1339.2093296
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2091391, upper bound: 1339.2093041
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.1921156, upper bound: 1339.1920614
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.1921151, upper bound: 1339.1920614
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2084435, upper bound: 1339.2080436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 3, lower bound: -1339.2080436, upper bound: 1339.2080924

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2077076, upper bound: 1339.2077086
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2077076, upper bound: 1339.2077086
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086331, upper bound: 1339.2089052
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085659, upper bound: 1339.2086071
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2057939, upper bound: 1339.2058774
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2057939, upper bound: 1339.2058976
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2063439, upper bound: 1339.2065933
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2063439, upper bound: 1339.2065933
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084272, upper bound: 1339.2087392
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084272, upper bound: 1339.2087339
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1955575, upper bound: 1339.1955575
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1955575, upper bound: 1339.1955575
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083629, upper bound: 1339.2083629
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083629, upper bound: 1339.2083629
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108583, upper bound: 1339.2108884
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107610, upper bound: 1339.2107610
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1755667, upper bound: 1339.1755667
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1755667, upper bound: 1339.1755667
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2093411, upper bound: 1339.2095179
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2093411, upper bound: 1339.2095014
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045977, upper bound: 1339.2046005
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045977, upper bound: 1339.2046005
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2053781, upper bound: 1339.2056269
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2053781, upper bound: 1339.2055031
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090665, upper bound: 1339.2093472
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090665, upper bound: 1339.2092552
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097514, upper bound: 1339.2097890
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097966, upper bound: 1339.2097695
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086446, upper bound: 1339.2086446
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086446, upper bound: 1339.2086446
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2029369, upper bound: 1339.2029369
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2029369, upper bound: 1339.2029369
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2016509, upper bound: 1339.2016509
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2016509, upper bound: 1339.2016509
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2017852, upper bound: 1339.2019599
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2017852, upper bound: 1339.2017852
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2014960, upper bound: 1339.2016075
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2014960, upper bound: 1339.2014960
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1969427, upper bound: 1339.1969427
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1969427, upper bound: 1339.1969427
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074673, upper bound: 1339.2076371
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074673, upper bound: 1339.2075873
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086089, upper bound: 1339.2091580
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086089, upper bound: 1339.2086089
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086459, upper bound: 1339.2089092
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086454, upper bound: 1339.2089092
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073292, upper bound: 1339.2073290
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073292, upper bound: 1339.2073290
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074850, upper bound: 1339.2074647
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074851, upper bound: 1339.2074647
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2048392, upper bound: 1339.2049641
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2047895, upper bound: 1339.2047895
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046
1: -994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672
2: -529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268
3: -910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723
4: -664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079843, upper bound: 1339.2080918
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079632, upper bound: 1339.2080691
time: 1.13 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.21 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2077076, upper bound: 1339.2077086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2077076, upper bound: 1339.2077086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086331, upper bound: 1339.2089052
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2085659, upper bound: 1339.2086071
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2057939, upper bound: 1339.2058774
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2057939, upper bound: 1339.2058976
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2063439, upper bound: 1339.2065933
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2063439, upper bound: 1339.2065933
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2085220, upper bound: 1339.2085220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2084272, upper bound: 1339.2087392
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2084272, upper bound: 1339.2087339
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1955575, upper bound: 1339.1955575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1955575, upper bound: 1339.1955575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2083629, upper bound: 1339.2083629
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2083629, upper bound: 1339.2083629
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2108583, upper bound: 1339.2108884
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2107610, upper bound: 1339.2107610
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1755667, upper bound: 1339.1755667
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1755667, upper bound: 1339.1755667
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2093411, upper bound: 1339.2095179
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2093411, upper bound: 1339.2095014
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2045977, upper bound: 1339.2046005
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2045977, upper bound: 1339.2046005
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2053781, upper bound: 1339.2056269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2053781, upper bound: 1339.2055031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2090665, upper bound: 1339.2093472
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2090665, upper bound: 1339.2092552
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2097514, upper bound: 1339.2097890
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2097966, upper bound: 1339.2097695
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086446, upper bound: 1339.2086446
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086446, upper bound: 1339.2086446
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2029369, upper bound: 1339.2029369
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2029369, upper bound: 1339.2029369
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2016509, upper bound: 1339.2016509
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2016509, upper bound: 1339.2016509
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2017852, upper bound: 1339.2019599
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2017852, upper bound: 1339.2017852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2014960, upper bound: 1339.2016075
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2014960, upper bound: 1339.2014960
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1969427, upper bound: 1339.1969427
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.1969427, upper bound: 1339.1969427
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2074673, upper bound: 1339.2076371
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2074673, upper bound: 1339.2075873
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086089, upper bound: 1339.2091580
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086089, upper bound: 1339.2086089
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086459, upper bound: 1339.2089092
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2086454, upper bound: 1339.2089092
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2073292, upper bound: 1339.2073290
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2073292, upper bound: 1339.2073290
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2074850, upper bound: 1339.2074647
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2074851, upper bound: 1339.2074647
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2048392, upper bound: 1339.2049641
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2047895, upper bound: 1339.2047895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2079843, upper bound: 1339.2080918
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.21
Output dim: 3, lower bound: -1339.2079632, upper bound: 1339.2080691
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086332
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2082955, upper bound: 1339.2086159
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2045631, upper bound: 1339.2045631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2135731
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2133306, upper bound: 1339.2136535
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2132980, upper bound: 1339.2135805
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2132979, upper bound: 1339.2136638
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2137122, upper bound: 1339.2136575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2137555, upper bound: 1339.2134516
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2136938, upper bound: 1339.2136938
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2079419
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2079419, upper bound: 1339.2081356
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2113487
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2113487, upper bound: 1339.2114700
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2123920, upper bound: 1339.2123920
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2104315
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2100379, upper bound: 1339.2100379
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2033820
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2033820, upper bound: 1339.2034120
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112437
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2112437, upper bound: 1339.2112460
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2112477, upper bound: 1339.2113689
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2082403
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2081402, upper bound: 1339.2081402
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2115141, upper bound: 1339.2114353
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2117688, upper bound: 1339.2114313
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2083075, upper bound: 1339.2083075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2089250, upper bound: 1339.2083075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2072714, upper bound: 1339.2072714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2073047, upper bound: 1339.2072714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2084957, upper bound: 1339.2084957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2085416, upper bound: 1339.2084957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2065138, upper bound: 1339.2064924
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2065007, upper bound: 1339.2064924
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2063318, upper bound: 1339.2061163
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2064164, upper bound: 1339.2061142
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2129620, upper bound: 1339.2129620
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2080224, upper bound: 1339.2080224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2091634, upper bound: 1339.2093296
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2091391, upper bound: 1339.2093041
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2078565, upper bound: 1339.2078222
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2034964, upper bound: 1339.2034964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2044885, upper bound: 1339.2044885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2081377, upper bound: 1339.2079336
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2084435, upper bound: 1339.2080436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 3, lower bound: -1339.2080436, upper bound: 1339.2080924

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.33 + 417.96 = 421.29 seconds
