## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 147.6105270206


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288)
1: (-23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952)
2: (-12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484)
3: (-17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907)
4: (-24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 1.75 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -147.9063397, upper bound: 147.9063397

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9058381, upper bound: 147.9058464
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9058381, upper bound: 147.9058381
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.84 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -147.9058381, upper bound: 147.9058464
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -147.9058381, upper bound: 147.9058381

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8893966, upper bound: 147.8893966
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8893966, upper bound: 147.8893966
time: 0.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9052846, upper bound: 147.9057057
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9052846, upper bound: 147.9052859
time: 0.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.56
Output dim: 0, lower bound: -147.8893966, upper bound: 147.8893966
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.56
Output dim: 0, lower bound: -147.8893966, upper bound: 147.8893966
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.56
Output dim: 0, lower bound: -147.9052846, upper bound: 147.9057057
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.56
Output dim: 0, lower bound: -147.9052846, upper bound: 147.9052859

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8837561, upper bound: 147.8836735
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8837561, upper bound: 147.8836735
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8892578, upper bound: 147.8892568
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8892568, upper bound: 147.8892578
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8850114, upper bound: 147.8850117
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8850114, upper bound: 147.8850117
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878760, upper bound: 147.8878660
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878660, upper bound: 147.8878660
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8837561, upper bound: 147.8836735
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8837561, upper bound: 147.8836735
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8892578, upper bound: 147.8892568
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8892568, upper bound: 147.8892578
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8850114, upper bound: 147.8850117
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8850114, upper bound: 147.8850117
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8878760, upper bound: 147.8878660
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.54
Output dim: 0, lower bound: -147.8878660, upper bound: 147.8878660

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820425, upper bound: 147.8819994
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838752
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838752
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838792
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838792
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8849713, upper bound: 147.8849713
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8849713, upper bound: 147.8849717
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8796036, upper bound: 147.8796036
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8796036, upper bound: 147.8796036
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878589, upper bound: 147.8878199
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878427, upper bound: 147.8878492
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8877006, upper bound: 147.8876897
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8876718, upper bound: 147.8876905
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8820425, upper bound: 147.8819994
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8819994, upper bound: 147.8819994
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838752
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838752
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838792
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8836620, upper bound: 147.8838792
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8849713, upper bound: 147.8849713
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8849713, upper bound: 147.8849717
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8796036, upper bound: 147.8796036
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8796036, upper bound: 147.8796036
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8878589, upper bound: 147.8878199
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8878427, upper bound: 147.8878492
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8877006, upper bound: 147.8876897
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.59
Output dim: 0, lower bound: -147.8876718, upper bound: 147.8876905

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748685, upper bound: 147.8748685
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748685, upper bound: 147.8748685
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817899, upper bound: 147.8817899
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817899, upper bound: 147.8817899
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838310
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838324
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8772688
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8772688
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838216
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838399
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8771659
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8771659
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8831924, upper bound: 147.8831924
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8831924, upper bound: 147.8831924
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8849297, upper bound: 147.8849297
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8849297, upper bound: 147.8849297
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8795436, upper bound: 147.8795436
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8795436, upper bound: 147.8795436
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8784645, upper bound: 147.8784645
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8784645, upper bound: 147.8784645
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8848282, upper bound: 147.8846845
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8848706, upper bound: 147.8846845
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8856951, upper bound: 147.8858862
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8856951, upper bound: 147.8856914
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8876835, upper bound: 147.8876445
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8876662, upper bound: 147.8876730
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8845010, upper bound: 147.8846476
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8845177, upper bound: 147.8845010
time: 0.67 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8748685, upper bound: 147.8748685
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8748685, upper bound: 147.8748685
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8819480, upper bound: 147.8819480
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8817899, upper bound: 147.8817899
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8817899, upper bound: 147.8817899
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838310
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838324
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8772688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8772688
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838216
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8836212, upper bound: 147.8838399
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8771659
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8771659, upper bound: 147.8771659
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8831924, upper bound: 147.8831924
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8831924, upper bound: 147.8831924
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8849297, upper bound: 147.8849297
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8849297, upper bound: 147.8849297
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8795436, upper bound: 147.8795436
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8795436, upper bound: 147.8795436
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8784645, upper bound: 147.8784645
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8784645, upper bound: 147.8784645
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8848282, upper bound: 147.8846845
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8848706, upper bound: 147.8846845
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8856951, upper bound: 147.8858862
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8856951, upper bound: 147.8856914
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8876835, upper bound: 147.8876445
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8876662, upper bound: 147.8876730
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8845010, upper bound: 147.8846476
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.84
Output dim: 0, lower bound: -147.8845177, upper bound: 147.8845010

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817915, upper bound: 147.8817386
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817386, upper bound: 147.8817386
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8835857, upper bound: 147.8835857
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8835857, upper bound: 147.8837913
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819328, upper bound: 147.8821389
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8819328, upper bound: 147.8819328
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8834102, upper bound: 147.8834111
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8834102, upper bound: 147.8836394
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8757427, upper bound: 147.8757427
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8757427, upper bound: 147.8757427
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769920, upper bound: 147.8769920
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8769920, upper bound: 147.8769920
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8819390
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8793659, upper bound: 147.8793659
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8793659, upper bound: 147.8793659
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8784046, upper bound: 147.8784046
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8784046, upper bound: 147.8784046
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746944, upper bound: 147.8746944
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746944, upper bound: 147.8746944
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844020, upper bound: 147.8842158
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843816, upper bound: 147.8842158
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8855237, upper bound: 147.8857176
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8855237, upper bound: 147.8857210
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8855662, upper bound: 147.8855627
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8855662, upper bound: 147.8855627
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8846123, upper bound: 147.8846040
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844855, upper bound: 147.8844855
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8840335, upper bound: 147.8840335
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8840335, upper bound: 147.8841801
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844359, upper bound: 147.8844156
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844156, upper bound: 147.8844156
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817915, upper bound: 147.8817386
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817386, upper bound: 147.8817386
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8835857, upper bound: 147.8835857
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8835857, upper bound: 147.8837913
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769400, upper bound: 147.8769400
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8819328, upper bound: 147.8821389
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8819328, upper bound: 147.8819328
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8834102, upper bound: 147.8834111
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8834102, upper bound: 147.8836394
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8757427, upper bound: 147.8757427
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8757427, upper bound: 147.8757427
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769920, upper bound: 147.8769920
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8769920, upper bound: 147.8769920
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8829870, upper bound: 147.8829870
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8818962
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8818962, upper bound: 147.8819390
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8793659, upper bound: 147.8793659
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8793659, upper bound: 147.8793659
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8748283, upper bound: 147.8748283
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8784046, upper bound: 147.8784046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8784046, upper bound: 147.8784046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8732457, upper bound: 147.8732457
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8746944, upper bound: 147.8746944
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8746944, upper bound: 147.8746944
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8844020, upper bound: 147.8842158
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8843816, upper bound: 147.8842158
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8855237, upper bound: 147.8857176
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8855237, upper bound: 147.8857210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8855662, upper bound: 147.8855627
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8855662, upper bound: 147.8855627
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8817381, upper bound: 147.8817381
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8846123, upper bound: 147.8846040
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8844855, upper bound: 147.8844855
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8840335, upper bound: 147.8840335
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8840335, upper bound: 147.8841801
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8844359, upper bound: 147.8844156
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -147.8844156, upper bound: 147.8844156

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730321, upper bound: 147.8730321
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730321, upper bound: 147.8730321
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818796, upper bound: 147.8818796
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818796, upper bound: 147.8818796
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8833750, upper bound: 147.8833750
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8833750, upper bound: 147.8835893
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817210, upper bound: 147.8817210
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8817210, upper bound: 147.8819328
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755480, upper bound: 147.8755480
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755480, upper bound: 147.8755480
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755689, upper bound: 147.8755689
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8755689, upper bound: 147.8755689
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829685, upper bound: 147.8829685
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8829685, upper bound: 147.8829685
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798651, upper bound: 147.8798651
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798651, upper bound: 147.8798651
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8817370
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8851068
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8853007
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852329
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852363
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8817756
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000
time: 0.45 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730321, upper bound: 147.8730321
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730321, upper bound: 147.8730321
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730114, upper bound: 147.8730114
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744216, upper bound: 147.8744216
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818796, upper bound: 147.8818796
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818796, upper bound: 147.8818796
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8833750, upper bound: 147.8833750
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8833750, upper bound: 147.8835893
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8817210, upper bound: 147.8817210
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8817210, upper bound: 147.8819328
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8745934, upper bound: 147.8745934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767477, upper bound: 147.8767477
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755150, upper bound: 147.8755150
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755480, upper bound: 147.8755480
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755480, upper bound: 147.8755480
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755689, upper bound: 147.8755689
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8755689, upper bound: 147.8755689
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8767661, upper bound: 147.8767661
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8829685, upper bound: 147.8829685
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8829685, upper bound: 147.8829685
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8798651, upper bound: 147.8798651
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8798651, upper bound: 147.8798651
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8781125, upper bound: 147.8781125
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8818811, upper bound: 147.8818811
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800085, upper bound: 147.8800085
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8817370
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8851068
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8853007
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8817756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.78
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8797984, upper bound: 147.8797984
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8797984, upper bound: 147.8797984
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744450, upper bound: 147.8744450
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744450, upper bound: 147.8744450
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816645, upper bound: 147.8816645
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8816645, upper bound: 147.8816645
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8801178
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753749, upper bound: 147.8753749
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753749, upper bound: 147.8753749
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8779351, upper bound: 147.8779351
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8779351, upper bound: 147.8779351
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
time: 0.42 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8797984, upper bound: 147.8797984
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8797984, upper bound: 147.8797984
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727983, upper bound: 147.8727983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8728259, upper bound: 147.8728259
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744450, upper bound: 147.8744450
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744450, upper bound: 147.8744450
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816695, upper bound: 147.8816695
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816645, upper bound: 147.8816645
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8816645, upper bound: 147.8816645
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8801178
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8743800, upper bound: 147.8743800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753159, upper bound: 147.8753159
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729693, upper bound: 147.8729693
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729980, upper bound: 147.8729980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753749, upper bound: 147.8753749
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753749, upper bound: 147.8753749
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8765747, upper bound: 147.8765747
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8753415, upper bound: 147.8753415
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8798498, upper bound: 147.8798498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8727966, upper bound: 147.8727966
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8779351, upper bound: 147.8779351
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8779351, upper bound: 147.8779351
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8780598, upper bound: 147.8780598
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8744191, upper bound: 147.8744191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8800001, upper bound: 147.8800001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8742474, upper bound: 147.8742474
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.95
Output dim: 0, lower bound: -147.8729933, upper bound: 147.8729933
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8791335, upper bound: 147.8791335
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8782953, upper bound: 147.8782953
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8782248, upper bound: 147.8782248
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8732064, upper bound: 147.8732064
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8730382, upper bound: 147.8730382
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8746541, upper bound: 147.8746541
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744593, upper bound: 147.8744593
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8851068
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8851068, upper bound: 147.8853007
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8798587, upper bound: 147.8798587
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8852363, upper bound: 147.8852363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8800602, upper bound: 147.8800602
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8816868, upper bound: 147.8816868
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8744851, upper bound: 147.8744851
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8817756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8813558, upper bound: 147.8813558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8839953, upper bound: 147.8839953
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.95
Output dim: 0, lower bound: -147.8844000, upper bound: 147.8844000

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.52 + 417.67 = 420.19 seconds
