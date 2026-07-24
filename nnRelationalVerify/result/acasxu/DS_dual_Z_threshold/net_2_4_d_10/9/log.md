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
execution time: IAR + RelationalAnalysis = 1.84 + 1.85 = 3.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -147.9063397, upper bound: 147.9063397

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9055652, upper bound: 147.9056220
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9055652, upper bound: 147.9055652
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -147.9055652, upper bound: 147.9056220
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -147.9055652, upper bound: 147.9055652

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9054831
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9047402
time: 0.57 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9054190
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9046707
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.40 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9054831
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9047402
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9054190
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -147.9046707, upper bound: 147.9046707

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9053230
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046067, upper bound: 147.9046163
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046053, upper bound: 147.9046473
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046053, upper bound: 147.9045998
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9052993
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9051911
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046067
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046053
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9053230
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9046067, upper bound: 147.9046163
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9046053, upper bound: 147.9046473
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9046053, upper bound: 147.9045998
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9052993
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9051911
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046067
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046053

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9053230
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9049147
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046090
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046067, upper bound: 147.9046163
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046180, upper bound: 147.9046123
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046473
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9045998
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9052993, upper bound: 147.9045998
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9052993
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046186
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9051911
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046180
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046067
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046090, upper bound: 147.9045998
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046053
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9045998
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9053230
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9049147
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046090
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9046067, upper bound: 147.9046163
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9046180, upper bound: 147.9046123
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046473
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9045998
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9052993, upper bound: 147.9045998
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9052993
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046186
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9051911
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046180
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046067
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9046090, upper bound: 147.9045998
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9046053
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -147.9045998, upper bound: 147.9045998

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052341
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9053094
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9048828
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045957, upper bound: 147.9048997
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045961
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046000
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046072
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045957, upper bound: 147.9046074
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046091, upper bound: 147.9045956
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046089, upper bound: 147.9046025
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045956
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046322
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046095, upper bound: 147.9045908
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045977, upper bound: 147.9045908
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052117
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052865
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046095
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046097
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9051514
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9051801
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046025, upper bound: 147.9046089
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045956, upper bound: 147.9046091
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045957
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046072, upper bound: 147.9045977
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045961, upper bound: 147.9045908
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9046322, upper bound: 147.9045957
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045964
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
time: 0.52 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9053094
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9048828
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045957, upper bound: 147.9048997
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045961
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046000
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046072
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045957, upper bound: 147.9046074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046091, upper bound: 147.9045956
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046089, upper bound: 147.9046025
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045956
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046095, upper bound: 147.9045908
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045977, upper bound: 147.9045908
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052117
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9052865
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046095
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9046097
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9051514
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9051801
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046025, upper bound: 147.9046089
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045956, upper bound: 147.9046091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045957
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046072, upper bound: 147.9045977
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045961, upper bound: 147.9045908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9046322, upper bound: 147.9045957
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -147.9045908, upper bound: 147.9045908

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862833
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862833
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862471, upper bound: 147.8862401
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862471, upper bound: 147.8862435
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862435
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862885
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862885
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862444, upper bound: 147.8862401
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862444
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862444
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862918
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862918
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862885, upper bound: 147.8862401
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862885, upper bound: 147.8862401
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862435, upper bound: 147.8862471
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862471
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862771
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862771
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862833, upper bound: 147.8862401
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
time: 0.48 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862471, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862471, upper bound: 147.8862435
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862435
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862885
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862885
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862444, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862444
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862444
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862918
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862918
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862885, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862885, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862435, upper bound: 147.8862471
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862471
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862771
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862771
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862833, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -147.8862401, upper bound: 147.8862401

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842011
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842011
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841866
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841866
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841984
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841984
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841851, upper bound: 147.8841833
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841851, upper bound: 147.8841833
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842134
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842134
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842334, upper bound: 147.8841833
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842334, upper bound: 147.8841833
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842134, upper bound: 147.8841833
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841851
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841851
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841984, upper bound: 147.8841833
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842290, upper bound: 147.8843069
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8842011, upper bound: 147.8841833
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
time: 0.88 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842011
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842011
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843483
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843069, upper bound: 147.8842290
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842798
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841866
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841866
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843574
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841984
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841984
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843694, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841851, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841851, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842134
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842134
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842334, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842334, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8842334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842134, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841851
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841851
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843694
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841984, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843574, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842798, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842290, upper bound: 147.8843069
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8843069
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841866, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8843483, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8842011, upper bound: 147.8841833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -147.8841833, upper bound: 147.8841833

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288
1: -23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952
2: -12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484
3: -17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907
4: -24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.69 + 416.37 = 420.06 seconds
