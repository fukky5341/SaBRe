## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 47.0393777385


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003)
1: (-10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643)
2: (-10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861)
3: (-15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287)
4: (-17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 1.47 = 2.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809205, upper bound: 47.1809205

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1670841
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1679708
time: 0.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1670841
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1679708
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1670841
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1679708
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1670841
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 4, lower bound: -47.1670841, upper bound: 47.1679708

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.42 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.04
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.26 + 27.96 = 30.22 seconds
