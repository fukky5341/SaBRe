## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 743.673742927666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447)
1: (-226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553)
2: (-160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006)
3: (-390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531)
4: (-263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.63 + 1.83 = 2.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -743.6886167, upper bound: 743.6886167

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6876140, upper bound: 743.6875272
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6875272, upper bound: 743.6876140
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.52 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -743.6876140, upper bound: 743.6875272
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -743.6875272, upper bound: 743.6876140

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6870089
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6870089
time: 0.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6871066
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6871066
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.65 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.65
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6870089
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.65
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6870089
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.65
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6871066
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.65
Output dim: 0, lower bound: -743.6870089, upper bound: 743.6871066

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447
1: -226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553
2: -160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006
3: -390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531
4: -263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6695560
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6696842
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6695560, upper bound: 743.6695560
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.78
Output dim: 0, lower bound: -743.6696842, upper bound: 743.6695560

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.46 + 12.11 = 14.57 seconds
