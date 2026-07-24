## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.42 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.48 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.47 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.63 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.91 + 71.40 = 74.31 seconds
