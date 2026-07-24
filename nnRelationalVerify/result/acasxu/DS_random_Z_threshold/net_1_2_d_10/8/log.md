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
execution time: IAR + RelationalAnalysis = 0.76 + 1.41 = 2.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5595995
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5595995
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5595995
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5595995

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.39 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5577271
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577271, upper bound: 43.5595995
time: 0.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.74
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.74
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.74
Output dim: 3, lower bound: -43.5595995, upper bound: 43.5577271
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.74
Output dim: 3, lower bound: -43.5577271, upper bound: 43.5595995

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5492901, upper bound: 43.5499616
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5492901, upper bound: 43.5499616
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.49 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5492901, upper bound: 43.5499616
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.49
Output dim: 3, lower bound: -43.5492901, upper bound: 43.5499616

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.66
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3640642
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3640642
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5490400
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497921
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.71 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3640642
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3640642
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5497921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5490400
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.71
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497921

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3774717
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3774717
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489811, upper bound: 43.5488756
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5490755, upper bound: 43.5497921
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490400
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5492503
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5497921
time: 0.47 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.16 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3774717
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3774717
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.4832400, upper bound: 43.4832400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5489811, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5490264, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5490755, upper bound: 43.5497921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490400
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5492503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5497921

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3817160
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3817160
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 46

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489811, upper bound: 43.5488756
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 29

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5475685
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5474871
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 40

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497207
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 29

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 40

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 40

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5482956, upper bound: 43.5477275
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5481953, upper bound: 43.5477275
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 46

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5490755, upper bound: 43.5488756
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488778, upper bound: 43.5497921
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5470284
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5439177
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 46

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 40

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478677
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5479818
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 29

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5475137, upper bound: 43.5477498
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 40

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 29

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473366
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5474432
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 29

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5492447
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489446, upper bound: 43.5497921
time: 0.42 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.92 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3817160
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3817160
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5489811, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5475685
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5474871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497207
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5482956, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5481953, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5490755, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488778, upper bound: 43.5497921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5470284
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5439177
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478677
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5479818
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5475137, upper bound: 43.5477498
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473366
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5474432
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5492447
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.92
Output dim: 3, lower bound: -43.5489446, upper bound: 43.5497921

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3653406
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3758455
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572985, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572985, upper bound: 43.3572016
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488838, upper bound: 43.5488756
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460253, upper bound: 43.5462050
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460160, upper bound: 43.5465081
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5474871
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472506
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5478035
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5478775
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5482956, upper bound: 43.5477275
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5468126, upper bound: 43.5464515
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469606, upper bound: 43.5464515
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490798
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488778, upper bound: 43.5497921
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5471592
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5478138, upper bound: 43.5477275
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478652
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478677
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5467139
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5465242
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5417408, upper bound: 43.5415493
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488984, upper bound: 43.5488756
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453899
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453110
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456657
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460761
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5491776
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5492447
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5480426
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5477310
time: 0.41 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.93 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3653406
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3758455
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572985, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572985, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3713640, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488838, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5460253, upper bound: 43.5462050
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5460160, upper bound: 43.5465081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5474871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472506
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5478035
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5478775
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5497664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5482956, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5468126, upper bound: 43.5464515
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469606, upper bound: 43.5464515
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490798
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488778, upper bound: 43.5497921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5471592
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5478138, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478652
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5478677
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5467139
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5465242
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5417408, upper bound: 43.5415493
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488984, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5491259, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453899
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460761
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5491776
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5492447
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5480426
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5477310

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3671719, upper bound: 43.3572016
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3592999, upper bound: 43.3572016
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470977, upper bound: 43.5470926
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471345, upper bound: 43.5470926
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5461450
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5462050
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438462
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440238, upper bound: 43.5439356
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5461860
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5458398
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472506
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452845, upper bound: 43.5459747
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5453394, upper bound: 43.5460716
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5480216
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5476640
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438500, upper bound: 43.5438500
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438500, upper bound: 43.5438500
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5468126, upper bound: 43.5464515
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470518, upper bound: 43.5464515
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5444944, upper bound: 43.5442041
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5445030, upper bound: 43.5442041
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472234, upper bound: 43.5470926
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471498, upper bound: 43.5470926
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477341
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477341
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477629
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477740, upper bound: 43.5479677
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5453652
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5453007
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5457383
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477975, upper bound: 43.5477275
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5460035
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5460035
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5459747
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5460765
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5460810
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5458860
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5466755
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5467139
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5424274, upper bound: 43.5424274
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5424274, upper bound: 43.5424274
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5411249, upper bound: 43.5411249
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5411249, upper bound: 43.5411249
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5392235, upper bound: 43.5392235
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5392235, upper bound: 43.5392235
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472629, upper bound: 43.5470926
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471278, upper bound: 43.5470926
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.45 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.96 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3671719, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3592999, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5470977, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5471345, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5461450
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5462050
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438462
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440238, upper bound: 43.5439356
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5461860
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5458398
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5469905
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472506
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452845, upper bound: 43.5459747
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5453394, upper bound: 43.5460716
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5480216
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5476640
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5474955, upper bound: 43.5474955
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5438500, upper bound: 43.5438500
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5438500, upper bound: 43.5438500
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5410108, upper bound: 43.5410108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5468126, upper bound: 43.5464515
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5470518, upper bound: 43.5464515
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444019, upper bound: 43.5444019
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5444944, upper bound: 43.5442041
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5445030, upper bound: 43.5442041
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5472234, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5472260, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5471498, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477341
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477341
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477629
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477740, upper bound: 43.5479677
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5426902, upper bound: 43.5426902
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5440209, upper bound: 43.5440209
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5453652
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5453007
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5457383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5430344, upper bound: 43.5430344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5415493, upper bound: 43.5415493
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457003, upper bound: 43.5457003
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477975, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5460035
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5460035, upper bound: 43.5460035
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5452674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5459747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5460765
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5460810
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5458860
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5466755
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5467139
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5424274, upper bound: 43.5424274
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5424274, upper bound: 43.5424274
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5455363, upper bound: 43.5455363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5411249, upper bound: 43.5411249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5411249, upper bound: 43.5411249
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5392235, upper bound: 43.5392235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5392235, upper bound: 43.5392235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5472629, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5471278, upper bound: 43.5470926
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.96
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453899
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460761
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5491776
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5489981, upper bound: 43.5492447
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5480426
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.96
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5477310

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.17 + 418.35 = 420.52 seconds
