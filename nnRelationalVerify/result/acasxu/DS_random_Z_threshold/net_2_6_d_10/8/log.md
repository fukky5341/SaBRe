## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.289434762380004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160)
1: (-17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844)
2: (-13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863)
3: (-14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786)
4: (-11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 1.59 = 2.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -42.3021254, upper bound: 42.3021254

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3010089, upper bound: 42.3009661
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3009661, upper bound: 42.3010089
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.93 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 3, lower bound: -42.3010089, upper bound: 42.3009661
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 3, lower bound: -42.3009661, upper bound: 42.3010089

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
time: 0.45 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.71 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.71
Output dim: 3, lower bound: -42.2903792, upper bound: 42.2903792

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875336, upper bound: 42.2875336
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875336, upper bound: 42.2875336
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880220, upper bound: 42.2880220
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880220, upper bound: 42.2880220
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870531, upper bound: 42.2870531
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870531, upper bound: 42.2870531
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.40 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2875336, upper bound: 42.2875336
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2875336, upper bound: 42.2875336
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2880220, upper bound: 42.2880220
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2880220, upper bound: 42.2880220
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2870531, upper bound: 42.2870531
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2870531, upper bound: 42.2870531
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2898427, upper bound: 42.2898427
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.64
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875286, upper bound: 42.2875286
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875286, upper bound: 42.2875286
time: 0.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2903744, upper bound: 42.2903744
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2875286, upper bound: 42.2875286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.74
Output dim: 3, lower bound: -42.2875286, upper bound: 42.2875286

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2885432, upper bound: 42.2885432
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2885432, upper bound: 42.2885432
time: 0.41 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2898374, upper bound: 42.2898374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2885432, upper bound: 42.2885432
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.90
Output dim: 3, lower bound: -42.2885432, upper bound: 42.2885432

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
time: 0.43 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2880167, upper bound: 42.2880167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2895329, upper bound: 42.2895329
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.74
Output dim: 3, lower bound: -42.2870477, upper bound: 42.2870477

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
time: 0.41 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2877077, upper bound: 42.2877077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 3, lower bound: -42.2867419, upper bound: 42.2867419

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.48 + 119.69 = 122.17 seconds
